"""
Módulo de Interface do Usuário (UI).
Versão Corrigida: Uso de Callbacks para evitar StreamlitAPIException.
"""

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
    Gerencia o layout, inputs do usuário e orquestração dos serviços.
    """

    def run(self) -> None:
        """Método principal de execução da interface (Entry Point)."""
        st.set_page_config(page_title="Icarus Valuation", layout="wide", page_icon="🛡️")
        st.title("🛡️ Icarus: Valuation IFRS 2 (Stable)")
        
        # --- Inicialização de Estado (Session State) ---
        if 'analysis_result' not in st.session_state:
            st.session_state['analysis_result'] = None
        if 'full_context_text' not in st.session_state:
            st.session_state['full_context_text'] = ""
        if 'tranches' not in st.session_state:
            st.session_state['tranches'] = []
        if 'mc_code' not in st.session_state:
            st.session_state['mc_code'] = ""

        # --- SIDEBAR: Upload e Análise ---
        with st.sidebar:
            st.header("1. Documentação")
            
            # Gestão de API Key
            if "GEMINI_API_KEY" in st.secrets:
                gemini_key = st.secrets["GEMINI_API_KEY"]
                st.success("🔑 API Key detectada (Secrets)")
            else:
                gemini_key = st.text_input("Gemini API Key", type="password")
            
            st.subheader("Upload de Contratos")
            uploaded_files = st.file_uploader(
                "PDF ou DOCX", 
                type=['pdf', 'docx'], 
                accept_multiple_files=True
            )
            
            manual_text = st.text_area(
                "Descrição Manual (Opcional)", 
                height=100, 
                placeholder="Cole cláusulas específicas aqui..."
            )
            
            if st.button("🚀 Analisar Contrato", type="primary"):
                self._handle_analysis(uploaded_files, manual_text, gemini_key)
            
            st.divider()
            st.caption("v.Release 1.2 - Callback Fix")

        # --- ÁREA PRINCIPAL ---
        if st.session_state['analysis_result']:
            self._render_dashboard(
                st.session_state['analysis_result'], 
                st.session_state['full_context_text'], 
                gemini_key
            )
        else:
            self._render_empty_state()

    # --- CALLBACKS (Novos métodos para correção do erro) ---
    def _update_widget_state(self, key_val: str, key_widget: str, value: float):
        """
        Callback genérico para atualizar o estado de um widget antes do rerun.
        Evita o erro 'StreamlitAPIException: Set value...'
        """
        st.session_state[key_val] = value
        st.session_state[key_widget] = value

    def _rate_search_callback(self, i: int, key_val: str, key_widget: str):
        """
        Callback específico para o botão de busca da Taxa DI.
        Executa a busca e atualiza o estado antes da renderização.
        """
        try:
            # Recupera as datas que estão nos widgets (dentro do popover)
            key_d1 = f"p_r_d1_{i}"
            key_d2 = f"p_r_d2_{i}"
            
            p_ref = st.session_state.get(key_d1, date.today())
            p_tgt = st.session_state.get(key_d2, date.today())
            
            # Executa a busca
            df_di = MarketDataService.get_di_data_b3(p_ref)
            if not df_di.empty:
                _, taxa, _ = MarketDataService.get_closest_di_vertex(p_tgt, df_di)
                new_rate = taxa * 100
                
                # Atualiza o estado
                st.session_state[key_val] = new_rate
                st.session_state[key_widget] = new_rate
                
                # Feedback visual (opcional, via toast pois st.success não renderiza bem em callback)
                st.toast(f"Taxa DI atualizada para {new_rate:.2f}%", icon="✅")
            else:
                st.toast("Não foi possível obter dados da B3.", icon="⚠️")
        except Exception as e:
            st.toast(f"Erro na busca: {str(e)}", icon="❌")

    # --- MÉTODOS DE RENDERIZAÇÃO ---
    def _render_empty_state(self):
        """Tela inicial antes da análise."""
        st.info("👈 Faça o upload do contrato na barra lateral para iniciar a análise.")
        st.markdown("""
        ### O que esta ferramenta faz?
        1.  **Lê Contratos:** Extrai cláusulas de Vesting, Lock-up e KPI de Performance.
        2.  **Classifica (IFRS 2):** Identifica se é *Equity-Settled* ou *Cash-Settled*.
        3.  **Dados de Mercado:** Busca Volatilidade (Yahoo) e Taxa DI (B3) automaticamente.
        4.  **Precificação:** Calcula o Fair Value usando Black-Scholes, Binomial ou Monte Carlo.
        """)

    def _handle_analysis(self, uploaded_files, manual_text: str, api_key: str) -> None:
        """Processa a entrada e chama o serviço de IA."""
        combined_text = ""
        
        if uploaded_files:
            with st.spinner("Lendo arquivos..."):
                for f in uploaded_files:
                    extracted = DocumentService.extract_text(f)
                    combined_text += f"--- {f.name} ---\n{extracted}\n"
        
        if manual_text:
            combined_text += f"--- MANUAL ---\n{manual_text}"
            
        if not combined_text.strip():
            st.error("⚠️ Forneça um arquivo ou texto manual.")
            return

        st.session_state['full_context_text'] = combined_text
        
        if api_key:
            with st.spinner("🤖 IA Analisando estrutura do plano..."):
                analysis = DocumentService.analyze_plan_with_gemini(combined_text, api_key)
        else:
            st.warning("⚠️ Sem API Key: Usando Mock.")
            analysis = DocumentService.mock_analysis(combined_text)
            
        if analysis:
            analysis = ModelSelectorService.select_model(analysis)
            st.session_state['analysis_result'] = analysis
            
            if analysis.tranches:
                st.session_state['tranches'] = [t for t in analysis.tranches]
            else:
                st.session_state['tranches'] = [
                    Tranche(vesting_date=1.0, proportion=1.0, expiration_date=analysis.option_life_years)
                ]

    def _render_dashboard(self, analysis: PlanAnalysisResult, full_text: str, api_key: str) -> None:
        """Renderiza o painel principal de valuation."""
        
        # --- Seção 1: Diagnóstico ---
        st.subheader("2. Diagnóstico e Parâmetros")
        
        with st.container():
            col_diag_main, col_diag_params = st.columns([1, 1])
            
            with col_diag_main:
                st.markdown("### 📋 Classificação IFRS 2")
                settlement = getattr(analysis, 'settlement_type', SettlementType.EQUITY_SETTLED)
                
                if settlement == SettlementType.CASH_SETTLED:
                    st.error(f"⚠️ **PASSIVO (Liability)** - {settlement.value}")
                    st.caption("Requer remensuração a cada balanço até a liquidação.")
                elif settlement == SettlementType.HYBRID:
                    st.warning(f"⚠️ **HÍBRIDO** - {settlement.value}")
                else:
                    st.success(f"✅ **EQUITY (Patrimônio)** - {settlement.value}")
                    st.caption("Mensurado na data de outorga (Grant Date).")

                st.markdown("**Justificativa Metodológica:**")
                st.info(analysis.methodology_rationale)

            with col_diag_params:
                st.markdown("### 🧮 Parâmetros de Valuation")
                val_params = getattr(analysis, 'valuation_params', None)
                prog_summary = getattr(analysis, 'program_summary', analysis.summary)
                
                if val_params and len(str(val_params)) > 10:
                    st.warning(val_params)
                else:
                    st.info(prog_summary)
                
                if val_params and len(str(val_params)) > 10:
                    with st.expander("Ver Resumo do Programa"):
                        st.write(prog_summary)

        st.divider()

        # --- Seção 2: Inputs Globais ---
        st.subheader("3. Premissas Globais (Ativo Objeto)")
        
        c_glob1, c_glob2, c_glob3, c_glob4 = st.columns(4)
        
        S = c_glob1.number_input("Preço da Ação (Spot) R$", 0.0, 10000.0, 50.0)
        K = c_glob2.number_input("Preço de Exercício (Strike) R$", 0.0, 10000.0, analysis.strike_price)
        q = c_glob3.number_input("Dividend Yield (% a.a.)", 0.0, 100.0, 4.0) / 100
        
        opts = [m for m in PricingModelType if m != PricingModelType.UNDEFINED]
        try: idx = opts.index(analysis.model_recommended)
        except ValueError: idx = 0
        active_model = c_glob4.selectbox("Modelo de Precificação", opts, index=idx)

        st.divider()

        # --- Seção 3: Configuração Granular por Tranche ---
        st.subheader("4. Configuração por Tranche")
        st.caption("Configure as premissas específicas. Use a lupa para buscar dados.")
        
        self._manage_tranches_buttons()
        tranches = st.session_state['tranches']
        
        calculation_inputs = []

        for i, t in enumerate(tranches):
            with st.container(border=True):
                c_head1, c_head2 = st.columns([1, 4])
                c_head1.markdown(f"#### Tranche {i+1}")
                
                def_exp = t.expiration_date if t.expiration_date else analysis.option_life_years
                
                col_t, col_vol, col_rate = st.columns([1.2, 1.5, 1.5])
                
                # A. Dados Temporais
                with col_t:
                    t_exp = st.number_input(f"Vencimento (Anos)", value=float(def_exp), min_value=0.01, step=0.1, key=f"t_exp_{i}")
                    t_vest = st.number_input(f"Vesting (Anos)", value=float(t.vesting_date), min_value=0.0, step=0.1, key=f"t_vest_{i}")
                    t_prop = st.number_input(f"Peso (%)", value=float(t.proportion * 100), step=1.0, key=f"t_prop_{i}") / 100

                # B. Volatilidade (Com Correção de State)
                with col_vol:
                    st.markdown("**Volatilidade**")
                    cv_input, cv_pop = st.columns([0.85, 0.15])
                    
                    key_vol_val = f"vol_val_{i}"
                    key_vol_widget = f"input_vol_{i}"
                    
                    if key_vol_val not in st.session_state: 
                        st.session_state[key_vol_val] = 30.00
                    
                    # Widget Principal
                    vol_final = cv_input.number_input(
                        "Anual (%)", 
                        value=st.session_state[key_vol_val], 
                        key=key_vol_widget,
                        format="%.2f", step=0.5, label_visibility="collapsed"
                    )
                    # Sincronia reversa (manual)
                    st.session_state[key_vol_val] = vol_final

                    # Popover Inteligente
                    with cv_pop.popover("🔍", help="Buscar Volatilidade Histórica"):
                        st.markdown("###### Calcular Volatilidade")
                        
                        p_tickers = st.text_area("Tickers", "VALE3", key=f"p_tk_{i}", height=68)
                        d_default_start = date.today() - timedelta(days=365*3)
                        p_start = st.date_input("Início", value=d_default_start, key=f"p_d1_{i}")
                        p_end = st.date_input("Fim", value=date.today(), key=f"p_d2_{i}")
                        
                        key_vol_search = f"vol_search_res_{i}"
                        
                        # Botão de Busca (Não conflitante)
                        if st.button("Buscar Dados", key=f"btn_calc_vol_{i}"):
                            ticker_list = [x.strip() for x in p_tickers.split(',') if x.strip()]
                            with st.spinner("Calculando..."):
                                res = MarketDataService.get_peer_group_volatility(ticker_list, p_start, p_end)
                                st.session_state[key_vol_search] = res
                        
                        # Exibição e Seleção
                        if key_vol_search in st.session_state:
                            res = st.session_state[key_vol_search]
                            if "summary" in res and res['summary']['count_valid'] > 0:
                                summ = res['summary']
                                st.success("Cálculo Sucesso!")
                                
                                opts_vol = {
                                    f"EWMA: {summ['mean_ewma']*100:.2f}%": summ['mean_ewma']*100,
                                    f"GARCH: {summ['mean_garch']*100:.2f}%": summ['mean_garch']*100,
                                    f"Histórica: {summ['mean_std']*100:.2f}%": summ['mean_std']*100
                                }
                                valid_opts = {k: v for k, v in opts_vol.items() if v > 0}
                                
                                if valid_opts:
                                    sel_label = st.radio("Métrica:", list(valid_opts.keys()), key=f"radio_vol_{i}")
                                    sel_val = valid_opts[sel_label]
                                    
                                    # BOTÃO COM CALLBACK (CORREÇÃO DO ERRO)
                                    st.button(
                                        "Aplicar Seleção", 
                                        key=f"btn_apply_vol_{i}",
                                        on_click=self._update_widget_state,
                                        args=(key_vol_val, key_vol_widget, sel_val)
                                    )
                                else:
                                    st.warning("Valores zerados.")
                            elif "details" in res:
                                st.error("Erro na busca.")

                # C. Taxa DI (Com Correção de State)
                with col_rate:
                    st.markdown("**Taxa Livre de Risco (DI)**")
                    cr_input, cr_pop = st.columns([0.85, 0.15])
                    
                    key_rate_val = f"rate_val_{i}"
                    key_rate_widget = f"input_rate_{i}"
                    
                    if key_rate_val not in st.session_state:
                        st.session_state[key_rate_val] = 10.75
                    
                    rate_final = cr_input.number_input(
                        "Anual (%)", 
                        value=st.session_state[key_rate_val], 
                        key=key_rate_widget,
                        format="%.2f", step=0.01, label_visibility="collapsed"
                    )
                    st.session_state[key_rate_val] = rate_final

                    with cr_pop.popover("📉", help="Buscar Taxa B3"):
                        st.markdown("###### Buscar Vértice DI")
                        p_ref = st.date_input("Data Base", value=date.today(), key=f"p_r_d1_{i}")
                        tgt = p_ref + timedelta(days=int(t_exp * 365))
                        p_tgt = st.date_input("Vencimento", value=tgt, key=f"p_r_d2_{i}")
                        
                        # BOTÃO COM CALLBACK (CORREÇÃO DO ERRO)
                        st.button(
                            "Buscar B3", 
                            key=f"btn_calc_rate_{i}",
                            on_click=self._rate_search_callback,
                            args=(i, key_rate_val, key_rate_widget)
                        )

                calculation_inputs.append({
                    "TrancheID": i+1,
                    "T": t_exp, "Vesting": t_vest, "Prop": t_prop,
                    "Vol": st.session_state[key_vol_val] / 100.0,
                    "r": st.session_state[key_rate_val] / 100.0,
                    "S": S, "K": K, "q": q, "Model": active_model,
                    "Lockup": analysis.lockup_years,
                    "M": analysis.early_exercise_multiple,
                    "Turnover": analysis.turnover_rate,
                    "StrikeCorr": 0.045 if analysis.has_strike_correction else 0.0
                })

        st.divider()

        if st.button("🏁 Calcular Fair Value Total", type="primary", use_container_width=True):
            self._execute_final_calculation(calculation_inputs)

    def _manage_tranches_buttons(self):
        c1, c2 = st.columns(2)
        if c1.button("➕ Adicionar Nova Tranche"):
            last_t = st.session_state['tranches'][-1] if st.session_state['tranches'] else None
            new_vest = (last_t.vesting_date + 1.0) if last_t else 1.0
            new_exp = (last_t.expiration_date) if last_t else 5.0
            st.session_state['tranches'].append(Tranche(vesting_date=new_vest, proportion=0.0, expiration_date=new_exp))
            st.rerun()
        
        if c2.button("➖ Remover Última"):
            if st.session_state['tranches']:
                st.session_state['tranches'].pop()
                st.rerun()

    def _execute_final_calculation(self, inputs: List[Dict]):
        results = []
        total_fv = 0.0
        
        st.markdown("### 📊 Resultados do Valuation")
        progress_bar = st.progress(0)
        
        for idx, item in enumerate(inputs):
            model = item['Model']
            fv_unit = 0.0
            try:
                if model == PricingModelType.BLACK_SCHOLES_GRADED:
                    fv_unit = FinancialMath.bs_call(
                        S=item['S'], K=item['K'], T=item['T'], 
                        r=item['r'], sigma=item['Vol'], q=item['q']
                    )
                elif model == PricingModelType.BINOMIAL:
                    fv_unit = FinancialMath.binomial_custom_optimized(
                        S=item['S'], K=item['K'], r=item['r'], vol=item['Vol'], q=item['q'],
                        vesting_years=item['Vesting'], turnover_w=item['Turnover'],
                        multiple_M=item['M'], hurdle_H=0.0, T_years=item['T'],
                        inflacao_anual=item['StrikeCorr'], lockup_years=item['Lockup']
                    )
                elif model == PricingModelType.RSU:
                    base_val = item['S'] * np.exp(-item['q'] * item['T'])
                    discount = 0.0
                    if item['Lockup'] > 0:
                        discount = FinancialMath.calculate_lockup_discount(
                            item['Vol'], item['Lockup'], base_val, item['q']
                        )
                    fv_unit = base_val - discount
                else:
                    fv_unit = FinancialMath.bs_call(
                        S=item['S'], K=item['K'], T=item['T'], 
                        r=item['r'], sigma=item['Vol'], q=item['q']
                    )
            except Exception as e:
                st.error(f"Erro no cálculo da Tranche {item['TrancheID']}: {e}")
                fv_unit = 0.0

            fv_weighted = fv_unit * item['Prop']
            total_fv += fv_weighted
            
            results.append({
                "Tranche": item['TrancheID'],
                "Vencimento": f"{item['T']:.2f} anos",
                "Volatilidade": f"{item['Vol']:.2%}",
                "Taxa Livre Risco": f"{item['r']:.2%}",
                "FV Unitário": fv_unit,
                "FV Ponderado": fv_weighted
            })
            progress_bar.progress((idx + 1) / len(inputs))

        c_res1, c_res2 = st.columns([1, 3])
        with c_res1:
            st.metric("Fair Value Total", f"R$ {total_fv:,.2f}")
        
        with c_res2:
            df_res = pd.DataFrame(results)
            st.dataframe(
                df_res.style.format({
                    "FV Unitário": "R$ {:.4f}", 
                    "FV Ponderado": "R$ {:.4f}"
                }),
                use_container_width=True, hide_index=True
            )
        
        csv = df_res.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Baixar Relatório (CSV)", data=csv, file_name="icarus_valuation_result.csv", mime="text/csv")
