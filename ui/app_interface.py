import streamlit as st
import pandas as pd
import numpy as np
import io
import sys
from datetime import date, timedelta
from typing import List, Dict

from core.domain import PlanAnalysisResult, Tranche, PricingModelType, SettlementType
from engines.financial import FinancialMath
from services.ai_service import DocumentService
from services.strategy import ModelSelectorService
from services.market_data import MarketDataService

class IFRS2App:
    """
    Aplicação principal do Icarus.
    """

    def run(self) -> None:
        st.set_page_config(page_title="Icarus Valuation", layout="wide", page_icon="🛡️")
        st.title("🛡️ Icarus: Valuation IFRS 2 (Table View)")
        
        if 'analysis_result' not in st.session_state:
            st.session_state['analysis_result'] = None
        if 'full_context_text' not in st.session_state:
            st.session_state['full_context_text'] = ""
        if 'tranches' not in st.session_state:
            st.session_state['tranches'] = []
        if 'mc_code' not in st.session_state:
            st.session_state['mc_code'] = ""

        with st.sidebar:
            st.header("1. Documentação")
            if "GEMINI_API_KEY" in st.secrets:
                gemini_key = st.secrets["GEMINI_API_KEY"]
                st.success("🔑 API Key detectada")
            else:
                gemini_key = st.text_input("Gemini API Key", type="password")
            
            st.subheader("Upload de Contratos")
            uploaded_files = st.file_uploader("PDF ou DOCX", type=['pdf', 'docx'], accept_multiple_files=True)
            manual_text = st.text_area("Descrição Manual", height=100, placeholder="Cole cláusulas aqui...")
            
            if st.button("🚀 Analisar Contrato", type="primary"):
                self._handle_analysis(uploaded_files, manual_text, gemini_key)
            
            st.divider()
            st.caption("v.Release 1.6 - DI Fix & UI Clean")

        if st.session_state['analysis_result']:
            self._render_dashboard(
                st.session_state['analysis_result'], 
                st.session_state['full_context_text'], 
                gemini_key
            )
        else:
            self._render_empty_state()

    def _update_widget_state(self, key_val: str, key_widget: str, value: float):
        st.session_state[key_val] = value
        st.session_state[key_widget] = value
        st.toast(f"Valor aplicado: {value:.4f}", icon="✅")

    def _render_empty_state(self):
        st.info("👈 Faça o upload do contrato para iniciar.")
        st.markdown("### Icarus Valuation\nFerramenta para precificação de opções (IFRS 2) com IA.")

    def _handle_analysis(self, uploaded_files, manual_text, api_key):
        combined_text = ""
        if uploaded_files:
            with st.spinner("Lendo arquivos..."):
                for f in uploaded_files:
                    combined_text += f"--- {f.name} ---\n{DocumentService.extract_text(f)}\n"
        if manual_text: combined_text += f"--- MANUAL ---\n{manual_text}"
        
        if not combined_text.strip():
            st.error("Forneça um arquivo ou texto.")
            return

        st.session_state['full_context_text'] = combined_text
        
        if api_key:
            with st.spinner("🤖 IA Analisando..."):
                analysis = DocumentService.analyze_plan_with_gemini(combined_text, api_key)
        else:
            st.warning("⚠️ Modo Mock (Sem API Key).")
            analysis = DocumentService.mock_analysis(combined_text)
            
        if analysis:
            analysis = ModelSelectorService.select_model(analysis)
            st.session_state['analysis_result'] = analysis
            if analysis.tranches:
                st.session_state['tranches'] = [t for t in analysis.tranches]
            else:
                st.session_state['tranches'] = [Tranche(1.0, 1.0, analysis.option_life_years)]

    def _render_dashboard(self, analysis: PlanAnalysisResult, full_text: str, api_key: str):
        st.subheader("2. Diagnóstico")
        with st.container():
            c1, c2 = st.columns(2)
            with c1:
                settlement = getattr(analysis, 'settlement_type', SettlementType.EQUITY_SETTLED)
                if settlement == SettlementType.CASH_SETTLED:
                    st.error(f"⚠️ PASSIVO (Liability) - {settlement.value}")
                else:
                    st.success(f"✅ EQUITY (Patrimônio) - {settlement.value}")
                st.info(analysis.methodology_rationale)
            with c2:
                st.markdown("**Parâmetros Extraídos:**")
                st.caption(getattr(analysis, 'valuation_params', analysis.summary))

        st.divider()

        st.subheader("3. Premissas de Mercado")
        c1, c2, c3, c4 = st.columns(4)
        S = c1.number_input("Spot (R$)", 0.0, 10000.0, 50.0)
        K = c2.number_input("Strike (R$)", 0.0, 10000.0, analysis.strike_price)
        q = c3.number_input("Div Yield (%)", 0.0, 100.0, 4.0) / 100
        
        opts = [m for m in PricingModelType if m != PricingModelType.UNDEFINED]
        idx = opts.index(analysis.model_recommended) if analysis.model_recommended in opts else 0
        active_model = c4.selectbox("Modelo", opts, index=idx)

        st.divider()

        st.subheader("4. Configuração e Cálculo")
        
        if active_model == PricingModelType.MONTE_CARLO:
            self._render_monte_carlo_ai(S, K, q, analysis, full_text, api_key)
        elif active_model == PricingModelType.BINOMIAL:
            self._render_binomial_graded(S, K, q, analysis)
        elif active_model == PricingModelType.BLACK_SCHOLES_GRADED:
            self._render_graded(S, K, q, analysis) 
        elif active_model == PricingModelType.RSU:
            self._render_rsu(S, q, analysis)

    def _render_binomial_graded(self, S, K, q, analysis):
        st.info("ℹ️ Modelo Binomial: Permite exercício antecipado, turnover e indexação de strike.")
        self._manage_tranches_buttons()
        tranches = st.session_state['tranches']
        inputs = []

        for i, t in enumerate(tranches):
            with st.container(border=True):
                st.markdown(f"**Tranche {i+1}**")
                col_t, col_vol, col_rate = st.columns([1, 1, 1])
                
                with col_t:
                    def_exp = t.expiration_date if t.expiration_date else analysis.option_life_years
                    t_exp = st.number_input(f"Vencimento (Anos)", value=float(def_exp), key=f"bi_t_{i}")
                    t_vest = st.number_input(f"Vesting (Anos)", value=float(t.vesting_date), key=f"bi_v_{i}")
                    t_prop = st.number_input(f"Peso (%)", value=float(t.proportion*100), key=f"bi_p_{i}")/100
                
                with col_vol:
                    self._render_vol_widget(i, "bi")
                    
                with col_rate:
                    self._render_rate_widget_table(i, "bi", t_exp)

                st.markdown("👇 *Parâmetros Avançados*")
                c_adv1, c_adv2, c_adv3, c_adv4 = st.columns(4)
                
                t_turnover = c_adv1.number_input(f"Turnover", value=float(analysis.turnover_rate * 100), key=f"bi_turn_{i}") / 100
                t_m = c_adv2.number_input(f"Múltiplo M", value=float(analysis.early_exercise_multiple), key=f"bi_m_{i}")
                t_strike_corr = c_adv3.number_input(f"Corr. Strike (%)", value=4.5 if analysis.has_strike_correction else 0.0, key=f"bi_corr_{i}") / 100
                t_lock = c_adv4.number_input(f"Lockup (Anos)", value=float(analysis.lockup_years), key=f"bi_lock_{i}")

                key_vol = f"vol_val_bi_{i}"
                key_rate = f"rate_val_bi_{i}"
                
                inputs.append({
                    "TrancheID": i+1, "S": S, "K": K, "q": q,
                    "T": t_exp, "Vesting": t_vest, "Prop": t_prop,
                    "Vol": st.session_state.get(key_vol, 0.30),
                    "r": st.session_state.get(key_rate, 0.1075),
                    "Turnover": t_turnover, "M": t_m, 
                    "StrikeCorr": t_strike_corr, "Lockup": t_lock
                })

        if st.button("Calcular (Binomial)", type="primary"):
            self._execute_calc(inputs, PricingModelType.BINOMIAL)

    def _render_monte_carlo_ai(self, S, K, q, analysis, text, api_key):
        st.warning("⚠️ Monte Carlo: Geração de Código via IA (Customizável)")
        tranches_dates = [t.vesting_date for t in st.session_state['tranches']]
        params = {
            "S0": S, "K": K, "q": q, "T": analysis.option_life_years,
            "vesting_schedule": tranches_dates,
            "barrier_type": "TSR/Performance" if analysis.has_market_condition else "None"
        }

        c1, c2 = st.columns([1, 1])
        if c1.button("1. Gerar Script Python (Gemini)"):
            with st.spinner("Escrevendo código..."):
                code = DocumentService.generate_custom_monte_carlo_code(text, params, api_key)
                st.session_state['mc_code'] = code
        
        if st.session_state['mc_code']:
            edited_code = st.text_area("Script", st.session_state['mc_code'], height=400)
            st.session_state['mc_code'] = edited_code
            if c2.button("2. Executar Simulação", type="primary"):
                self._run_custom_code(edited_code)

    def _render_graded(self, S, K, q, analysis):
        self._manage_tranches_buttons()
        tranches = st.session_state['tranches']
        inputs = []
        
        for i, t in enumerate(tranches):
            with st.container(border=True):
                st.markdown(f"**Tranche {i+1}**")
                c1, c2, c3 = st.columns(3)
                with c1:
                    def_exp = t.expiration_date if t.expiration_date else analysis.option_life_years
                    t_exp = st.number_input(f"Vencimento (T)", value=float(def_exp), key=f"bs_t_{i}")
                    t_prop = st.number_input(f"Peso (%)", value=float(t.proportion*100), key=f"bs_p_{i}")/100
                    t_vest = st.number_input(f"Vesting (Ref)", value=float(t.vesting_date), key=f"bs_v_{i}")
                with c2:
                    self._render_vol_widget(i, "bs")
                with c3:
                    self._render_rate_widget_table(i, "bs", t_exp)

                key_vol = f"vol_val_bs_{i}"
                key_rate = f"rate_val_bs_{i}"

                inputs.append({
                    "TrancheID": i+1, "S": S, "K": K, "q": q,
                    "T": t_exp, "Prop": t_prop,
                    "Vol": st.session_state.get(key_vol, 0.30),
                    "r": st.session_state.get(key_rate, 0.1075)
                })

        if st.button("Calcular (Black-Scholes)", type="primary"):
            self._execute_calc(inputs, PricingModelType.BLACK_SCHOLES_GRADED)

    def _render_rsu(self, S, q, analysis):
        self._manage_tranches_buttons()
        tranches = st.session_state['tranches']
        inputs = []
        
        for i, t in enumerate(tranches):
            with st.container(border=True):
                st.markdown(f"**Tranche {i+1}**")
                c1, c2, c3 = st.columns(3)
                t_vest = c1.number_input(f"Pagamento (Anos)", value=float(t.vesting_date), key=f"rsu_v_{i}")
                t_prop = c2.number_input(f"Peso (%)", value=float(t.proportion*100), key=f"rsu_p_{i}")/100
                t_lock = c3.number_input(f"Lockup (Anos)", value=float(analysis.lockup_years), key=f"rsu_l_{i}")
                
                vol_val = 0.30
                if t_lock > 0:
                    st.caption("Volatilidade para Lockup:")
                    self._render_vol_widget(i, "rsu")
                    vol_val = st.session_state.get(f"vol_val_rsu_{i}", 0.30)
                
                inputs.append({
                    "TrancheID": i+1, "S": S, "q": q, "T": t_vest, 
                    "Prop": t_prop, "Lockup": t_lock, "Vol": vol_val
                })
        
        if st.button("Calcular (RSU)", type="primary"):
            self._execute_calc(inputs, PricingModelType.RSU)

    def _render_vol_widget(self, i, prefix):
        st.markdown("Volatilidade (%)")
        c_in, c_pop = st.columns([0.85, 0.15])
        
        key_val = f"vol_val_{prefix}_{i}"
        key_w = f"vol_w_{prefix}_{i}"
        
        if key_val not in st.session_state: st.session_state[key_val] = 30.00
        
        val = c_in.number_input("Vol", value=st.session_state[key_val], key=key_w, label_visibility="collapsed", step=0.5)
        st.session_state[key_val] = val
        
        with c_pop.popover("🔍"):
            st.markdown("###### Calcular Volatilidade")
            tk = st.text_area("Tickers", "VALE3", key=f"tk_{prefix}_{i}")
            d1 = st.date_input("Ini", date.today()-timedelta(days=1000), key=f"d1_{prefix}_{i}")
            d2 = st.date_input("Fim", date.today(), key=f"d2_{prefix}_{i}")
            
            k_res = f"res_{prefix}_{i}"
            if st.button("Buscar", key=f"b_v_{prefix}_{i}"):
                with st.spinner("..."):
                    res = MarketDataService.get_peer_group_volatility([t.strip() for t in tk.split(',')], d1, d2)
                    st.session_state[k_res] = res
            
            if k_res in st.session_state:
                res = st.session_state[k_res]
                if "summary" in res:
                    summ = res['summary']
                    opts = {f"EWMA: {summ['mean_ewma']*100:.1f}%": summ['mean_ewma']*100,
                            f"Hist: {summ['mean_std']*100:.1f}%": summ['mean_std']*100}
                    sel = st.radio("Métrica", list(opts.keys()), key=f"rad_{prefix}_{i}")
                    
                    st.button("Aplicar", key=f"app_{prefix}_{i}", 
                              on_click=self._update_widget_state, args=(key_val, key_w, opts[sel]))

    def _render_rate_widget_table(self, i, prefix, t_years):
        """
        WIDGET DE DI OTIMIZADO: Callbacks Seguros + Tabela Limpa + Fix Warnings.
        """
        st.markdown("Taxa DI (%)")
        c_in, c_pop = st.columns([0.85, 0.15])
        
        key_val = f"rate_val_{prefix}_{i}"
        key_w = f"rate_w_{prefix}_{i}"
        
        if key_val not in st.session_state: 
            st.session_state[key_val] = 10.75
        
        # Sincroniza visual com estado
        current_pct = st.session_state[key_val] * 100
        val = c_in.number_input(
            "Rate", 
            value=float(current_pct), 
            key=key_w, 
            label_visibility="collapsed", 
            step=0.1,
            format="%.2f"
        )
        
        # Atualiza se input manual mudar
        new_decimal = val / 100.0
        if abs(new_decimal - st.session_state[key_val]) > 1e-6:
             st.session_state[key_val] = new_decimal

        with c_pop.popover("📉"):
            st.markdown("###### Consulta DI Futuro (B3)")
            
            d_base = st.date_input(
                "Data Base", 
                date.today(), 
                key=f"db_{prefix}_{i}",
                format="DD/MM/YYYY" 
            )
            
            k_df = f"df_di_{prefix}_{i}"
            
            if st.button("Buscar Taxas", key=f"b_load_di_{prefix}_{i}"):
                with st.spinner("Consultando B3..."):
                    df = MarketDataService.get_di_data_b3(d_base)
                    st.session_state[k_df] = df
            
            if k_df in st.session_state and not st.session_state[k_df].empty:
                df = st.session_state[k_df]
                
                # VISUALIZAÇÃO
                df_show = df.copy()
                df_show['Taxa (%)'] = (df_show['Taxa'] * 100).map('{:.2f}'.format)
                
                col_venc = 'Vencimento_Fmt' if 'Vencimento_Fmt' in df.columns else 'Vencimento_Str'
                
                st.caption("Taxas Disponíveis:")
                st.dataframe(
                    df_show[[col_venc, 'Taxa (%)']].rename(columns={col_venc: 'Vencimento'}), 
                    width='stretch',  # CORREÇÃO DO WARNING AQUI
                    height=200,
                    hide_index=True
                )
                
                # SELETOR
                df['Label'] = df.apply(
                    lambda x: f"{x[col_venc]} - {x['Taxa']*100:.2f}%", 
                    axis=1
                )
                
                target_days = t_years * 365
                idx_closest = (df['Dias_Corridos'] - target_days).abs().idxmin()
                
                st.markdown("**Selecione:**")
                selected_label = st.selectbox(
                    "Vencimento", 
                    options=df['Label'],
                    index=int(idx_closest),
                    key=f"sel_di_{prefix}_{i}",
                    label_visibility="collapsed"
                )
                
                # CALLBACK SEGURO (Atualiza decimal e visual)
                def apply_callback(k_decimal, k_widget, taxa_decimal):
                    st.session_state[k_decimal] = taxa_decimal
                    st.session_state[k_widget] = taxa_decimal * 100.0
                
                if selected_label:
                    row = df[df['Label'] == selected_label].iloc[0]
                    sel_taxa = row['Taxa']
                    
                    st.button(
                        f"Usar {selected_label}", 
                        key=f"b_apply_di_{prefix}_{i}",
                        on_click=apply_callback,
                        args=(key_val, key_w, sel_taxa)
                    )
            
            elif k_df in st.session_state:
                st.error("Nenhum dado encontrado para esta data.")

    def _run_custom_code(self, code):
        old_stdout = io.StringIO()
        sys.stdout = old_stdout
        local_scope = {}
        try:
            exec(code, local_scope)
            output = old_stdout.getvalue()
            sys.stdout = sys.__stdout__
            st.text("Saída do Script:")
            st.code(output)
            if 'fv' in local_scope:
                st.metric("Fair Value (fv)", f"R$ {local_scope['fv']:,.2f}")
        except Exception as e:
            sys.stdout = sys.__stdout__
            st.error(f"Erro: {e}")

    def _execute_calc(self, inputs, model_type):
        res_data = []
        total_fv = 0.0
        prog = st.progress(0)
        
        for idx, item in enumerate(inputs):
            try:
                if model_type == PricingModelType.BINOMIAL:
                    fv = FinancialMath.binomial_custom_optimized(
                        item['S'], item['K'], item['r'], item['Vol'], item['q'],
                        item['Vesting'], item['Turnover'], item['M'], 0.0,
                        item['T'], item['StrikeCorr'], item['Lockup']
                    )
                elif model_type == PricingModelType.BLACK_SCHOLES_GRADED:
                    fv = FinancialMath.bs_call(item['S'], item['K'], item['T'], item['r'], item['Vol'], item['q'])
                elif model_type == PricingModelType.RSU:
                    base = item['S'] * np.exp(-item['q']*item['T'])
                    disc = FinancialMath.calculate_lockup_discount(item['Vol'], item['Lockup'], base, item['q']) if item['Lockup'] > 0 else 0
                    fv = base - disc
                
                w_fv = fv * item['Prop']
                total_fv += w_fv
                
                res_data.append({
                    "Tranche": item['TrancheID'],
                    "FV Unit": fv,
                    "FV Ponderado": w_fv,
                    "Detalhes": str({k:v for k,v in item.items() if k not in ['S','K','r','Vol','q']})
                })
            except Exception as e:
                st.error(f"Erro Tranche {idx}: {e}")
            prog.progress((idx+1)/len(inputs))
            
        c1, c2 = st.columns([1,3])
        c1.metric("Fair Value Total", f"R$ {total_fv:,.2f}")
        c2.dataframe(pd.DataFrame(res_data))

    def _manage_tranches_buttons(self):
        c1, c2 = st.columns(2)
        if c1.button("➕ Adicionar Tranche"):
            st.session_state['tranches'].append(Tranche(1.0, 0.0, 5.0))
            st.rerun()
        if c2.button("➖ Remover Tranche"):
            if st.session_state['tranches']: st.session_state['tranches'].pop()
            st.rerun()
