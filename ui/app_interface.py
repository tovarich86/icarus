"""
Módulo de Interface do Usuário (UI).
Versão Final: Integração Completa (Market Panel + Cálculo por Tranche Automatizado).
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
    Aplicação principal do Icarus. Gerencia o estado da sessão e o layout.
    """

    def run(self) -> None:
        """Método principal de execução da interface."""
        st.title("🛡️ Icarus: Beta Modular")
        
        # Inicialização de Estado (Session State)
        if 'analysis_result' not in st.session_state:
            st.session_state['analysis_result'] = None
        if 'full_context_text' not in st.session_state:
            st.session_state['full_context_text'] = ""
        if 'tranches' not in st.session_state:
            st.session_state['tranches'] = []
        if 'mc_code' not in st.session_state:
            st.session_state['mc_code'] = ""
            
        # --- Cache de Dados de Mercado ---
        # Armazena os resultados da busca em lote (DI e Volatilidade)
        if 'market_cache' not in st.session_state:
            st.session_state['market_cache'] = {
                "di_curve": pd.DataFrame(),
                "vol_data": {},
                "ref_date": date.today(),
                "has_data": False
            }

        # --- SIDEBAR: Inputs ---
        with st.sidebar:
            st.header("Entradas")
            
            # Gestão de API Key
            if "GEMINI_API_KEY" in st.secrets:
                gemini_key = st.secrets["GEMINI_API_KEY"]
                st.success("🔑 API Key detectada (Secrets)")
            else:
                gemini_key = st.text_input("Gemini API Key", type="password")
            
            st.subheader("Dados do Plano")
            uploaded_files = st.file_uploader(
                "1. Upload de Contratos (PDF/DOCX)", 
                type=['pdf', 'docx'], 
                accept_multiple_files=True
            )
            
            manual_text = st.text_area(
                "2. Descrição Manual (Opcional)", 
                height=150, 
                placeholder="Cole trechos do contrato aqui..."
            )
            
            if st.button("🚀 Analisar Plano", type="primary"):
                self._handle_analysis(uploaded_files, manual_text, gemini_key)
            
            st.divider()
            st.caption("v.RC 1.0 - Full Integration")

        # --- ÁREA PRINCIPAL ---
        if st.session_state['analysis_result']:
            self._render_dashboard(
                st.session_state['analysis_result'], 
                st.session_state['full_context_text'], 
                gemini_key
            )
        else:
            st.info("👈 Por favor, forneça o contrato ou descrição na barra lateral para iniciar.")

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
            with st.spinner("🤖 IA Analisando estrutura do plano e classificação contábil..."):
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
        """Renderiza os resultados da análise e as calculadoras."""
        
        # --- Seção 1: Diagnóstico ---
        st.subheader("1. Diagnóstico e Classificação Contábil")
        
        settlement = getattr(analysis, 'settlement_type', SettlementType.EQUITY_SETTLED)
        if settlement == SettlementType.CASH_SETTLED:
            st.error(f"⚠️ **PASSIVO (Liability)** - {settlement.value}. Requer remensuração.")
        elif settlement == SettlementType.HYBRID:
            st.warning(f"⚠️ **HÍBRIDO** - {settlement.value}.")
        else:
            st.success(f"✅ **EQUITY (Patrimônio)** - {settlement.value}. Mensurado na Outorga.")

        with st.expander("Detalhes do Diagnóstico", expanded=False):
            c1, c2 = st.columns(2)
            c1.info(getattr(analysis, 'program_summary', analysis.summary))
            c2.warning(getattr(analysis, 'valuation_params', "N/A"))
            st.write(analysis.methodology_rationale)

        st.divider()

        # --- SEÇÃO 2: Configuração de Mercado e Peer Group ---
        st.subheader("2. Configuração de Mercado e Peer Group")
        
        with st.container(border=True):
            col_dates, col_tickers = st.columns([1, 2])
            
            with col_dates:
                st.markdown("###### 📅 Datas de Referência")
                ref_date = st.date_input(
                    "Data Base da Avaliação (Grant/Reporting)", 
                    value=date.today(),
                    help="Data para captura da curva de juros (DI) e data final para volatilidade."
                )
                
                st.markdown("###### 📊 Janela de Volatilidade")
                default_start = ref_date - timedelta(days=365*3) 
                vol_start = st.date_input("Início da Amostra", value=default_start)
                vol_end = st.date_input("Fim da Amostra", value=ref_date)

            with col_tickers:
                st.markdown("###### 🏢 Peer Group (Volatilidade)")
                st.caption("Insira os tickers das empresas comparáveis separados por vírgula.")
                tickers_input = st.text_area("Tickers (ex: VALE3, PETR4, CMIN3)", value="VALE3", height=107)

            # Botão de Ação Global
            if st.button("🔄 Carregar Dados de Mercado (DI & Volatilidade)", type="primary"):
                ticker_list = [t.strip() for t in tickers_input.split(',') if t.strip()]
                
                if not ticker_list:
                    st.error("Insira pelo menos um ticker.")
                else:
                    with st.spinner(f"Buscando curva DI de {ref_date.strftime('%d/%m/%Y')} e volatilidade..."):
                        # 1. Busca Volatilidade (Peer Group)
                        vol_result = MarketDataService.get_peer_group_volatility(
                            tickers=ticker_list,
                            start_date=vol_start,
                            end_date=vol_end
                        )
                        # 2. Busca Curva DI (Data Base)
                        di_curve = MarketDataService.get_di_curve(ref_date)
                        
                        # 3. Salva no Estado
                        st.session_state['market_cache'] = {
                            "di_curve": di_curve,
                            "vol_data": vol_result,
                            "ref_date": ref_date,
                            "has_data": True
                        }
                        
                        if "summary" in vol_result:
                            st.toast(f"Volatilidade Média: {vol_result['summary']['mean_ewma']:.2%}")
                        if not di_curve.empty:
                            st.toast("Curva DI carregada!")
                        else:
                            st.toast("⚠️ Aviso: Curva DI não encontrada (Feriado?).", icon="⚠️")

            # --- Visualização dos Dados Carregados ---
            cache = st.session_state['market_cache']
            if cache['has_data']:
                st.divider()
                exp_res = st.expander("Visualizar Dados Carregados", expanded=True)
                with exp_res:
                    c_vol, c_di = st.columns(2)
                    with c_vol:
                        st.markdown("**Volatilidade do Peer Group**")
                        vol_data = cache['vol_data']
                        if "summary" in vol_data:
                            summ = vol_data['summary']
                            m1, m2, m3 = st.columns(3)
                            m1.metric("Média StdDev", f"{summ['mean_std']:.2%}")
                            m2.metric("Média EWMA", f"{summ['mean_ewma']:.2%}")
                            m3.metric("Média GARCH", f"{summ['mean_garch']:.2%}")
                            
                            details = []
                            for t, d in vol_data['details'].items():
                                if "error" in d:
                                    details.append({"Ticker": t, "Erro": d['error']})
                                else:
                                    details.append({
                                        "Ticker": t,
                                        "StdDev": f"{d['std_dev']:.2%}",
                                        "EWMA": f"{d['ewma']:.2%}",
                                        "GARCH": f"{d['garch']:.2%}" if d['garch'] else "N/A"
                                    })
                            st.dataframe(pd.DataFrame(details), hide_index=True, use_container_width=True)

                    with c_di:
                        st.markdown(f"**Curva de Juros (B3) - {cache['ref_date'].strftime('%d/%m/%Y')}**")
                        df_di = cache['di_curve']
                        if not df_di.empty:
                            st.line_chart(df_di, x='anos', y='taxa', height=250)
                        else:
                            st.warning("Sem dados de DI.")

        st.divider()

        # --- SEÇÃO 3: Parâmetros de Precificação (Ativos) ---
        st.subheader("3. Parâmetros de Precificação")
        
        # Seleção de Modelo
        opts = [m for m in PricingModelType if m != PricingModelType.UNDEFINED]
        try: idx = opts.index(analysis.model_recommended)
        except ValueError: idx = 0
        active_model = st.selectbox("Modelo Selecionado:", opts, index=idx)

        # Inputs Globais Básicos (Spot, Strike, Yield)
        c_glob1, c_glob2, c_glob3 = st.columns(3)
        S = c_glob1.number_input("Preço da Ação (Spot) R$", 0.0, 10000.0, 50.0)
        K = c_glob2.number_input("Preço de Exercício (Strike) R$", 0.0, 10000.0, analysis.strike_price)
        q = c_glob3.number_input("Dividend Yield (% a.a.)", 0.0, 100.0, 4.0) / 100

        # --- SEÇÃO 4: Cálculo do Fair Value (Por Tranche) ---
        st.subheader("4. Cálculo do Fair Value (Por Tranche)")

        # Recupera dados de mercado do cache para autropreenchimento
        market_defaults = self._get_market_defaults()

        # Roteamento de Modelos
        if active_model == PricingModelType.BLACK_SCHOLES_GRADED:
            self._render_graded(S, K, q, analysis, market_defaults)
        elif active_model == PricingModelType.BINOMIAL:
            self._render_binomial_graded(S, K, q, analysis, market_defaults)
        elif active_model == PricingModelType.MONTE_CARLO:
            self._render_monte_carlo_ai(S, K, q, analysis, st.session_state['full_context_text'], api_key, market_defaults)
        elif active_model == PricingModelType.RSU:
            self._render_rsu(S, q, analysis, market_defaults)

    def _get_market_defaults(self) -> Dict:
        """
        Helper para extrair valores padrão (volatilidade média e curva DI) do cache.
        """
        defaults = {"vol": 0.30, "di_curve": pd.DataFrame(), "auto": False}
        
        cache = st.session_state.get('market_cache', {})
        if cache.get('has_data'):
            defaults["auto"] = True
            defaults["di_curve"] = cache['di_curve']
            
            # Preferência de Volatilidade: EWMA -> GARCH -> StdDev
            vol_summ = cache['vol_data'].get('summary', {})
            if vol_summ.get('mean_ewma', 0) > 0:
                defaults["vol"] = vol_summ['mean_ewma']
            elif vol_summ.get('mean_garch', 0) > 0:
                defaults["vol"] = vol_summ['mean_garch']
            elif vol_summ.get('mean_std', 0) > 0:
                defaults["vol"] = vol_summ['mean_std']
                
        return defaults

    def _manage_tranches(self) -> None:
        """Widget auxiliar para adicionar/remover tranches."""
        st.markdown("#### ⚙️ Gerenciar Tranches")
        c1, c2 = st.columns(2)
        if c1.button("➕ Adicionar Tranche"):
            last_tranche = st.session_state['tranches'][-1] if st.session_state['tranches'] else None
            new_vest = (last_tranche.vesting_date + 1.0) if last_tranche else 1.0
            new_exp = (last_tranche.expiration_date) if last_tranche else 10.0
            st.session_state['tranches'].append(Tranche(vesting_date=new_vest, proportion=0.0, expiration_date=new_exp))
            st.rerun()
        if c2.button("➖ Remover Última"):
            if len(st.session_state['tranches']) > 0:
                st.session_state['tranches'].pop()
                st.rerun()

    def _render_graded(self, S, K, q, analysis, mkt_defaults):
        st.info("ℹ️ Black-Scholes (Graded): Calcula cada tranche como uma opção independente.")
        if mkt_defaults['auto']:
            st.success("🟢 Dados de Mercado Carregados: Volatilidade Média e Taxas DI Interpoladas aplicadas automaticamente.")
        
        self._manage_tranches()
        tranches = st.session_state['tranches']
        if not tranches: return

        inputs = []
        st.markdown("---")
        
        for i, t in enumerate(tranches):
            with st.expander(f"Tranche {i+1}", expanded=True):
                # Prazo (T)
                def_exp = t.expiration_date if t.expiration_date else analysis.option_life_years
                
                # Autopreenchimento da Taxa DI
                t_r = 0.1075
                if not mkt_defaults['di_curve'].empty:
                    # Interpola a taxa para o prazo exato desta tranche
                    t_r = MarketDataService.interpolate_di_rate(def_exp, mkt_defaults['di_curve'])

                c1, c2, c3 = st.columns(3)
                t_exp = c1.number_input(f"Vencimento (T) {i}", value=float(def_exp), min_value=0.01, step=0.1, key=f"bs_t_{i}")
                
                label_vol = f"Volatilidade % {'(Auto)' if mkt_defaults['auto'] else ''}"
                t_vol = c2.number_input(label_vol, value=float(mkt_defaults['vol']*100), key=f"bs_v_{i}", format="%.2f") / 100
                
                label_r = f"Taxa Livre Risco % {'(Auto)' if mkt_defaults['auto'] else ''}"
                t_rate = c3.number_input(label_r, value=float(t_r*100), key=f"bs_r_{i}", format="%.2f", help="Taxa interpolada da curva DI") / 100
                
                c4, c5 = st.columns(2)
                t_vest = c4.number_input(f"Vesting (Ref) {i}", value=float(t.vesting_date), key=f"bs_ve_{i}")
                t_prop = c5.number_input(f"Peso % {i}", value=float(t.proportion*100), key=f"bs_p_{i}")/100
                
                inputs.append({"T": t_exp, "r": t_rate, "vol": t_vol, "prop": t_prop, "vest": t_vest})

        if st.button("Calcular (Black-Scholes)", type="primary"):
            total_fv = 0.0
            res = []
            for idx, item in enumerate(inputs):
                fv = FinancialMath.bs_call(S, K, item["T"], item["r"], item["vol"], q)
                w_fv = fv * item["prop"]
                total_fv += w_fv
                res.append({
                    "Tranche": idx+1,
                    "Vesting": item["vest"],
                    "Vencimento (T)": item["T"],
                    "Volatilidade": f"{item['vol']:.2%}",
                    "Taxa (r)": f"{item['r']:.2%}",
                    "FV Unitário": fv,
                    "FV Ponderado": w_fv
                })
            
            st.metric("Fair Value Total", f"R$ {total_fv:.4f}")
            st.dataframe(pd.DataFrame(res))

    def _render_binomial_graded(self, S, K, q, analysis, mkt_defaults):
        st.info("ℹ️ Modelo Lattice Binomial (Suporta Exercício Antecipado e Lock-up)")
        if mkt_defaults['auto']:
            st.success("🟢 Dados de Mercado Carregados.")

        self._manage_tranches()
        tranches = st.session_state['tranches']
        inputs = []

        for i, t in enumerate(tranches):
            with st.expander(f"Tranche {i+1}", expanded=False):
                # Definição de defaults automáticos
                def_exp = t.expiration_date if t.expiration_date else analysis.option_life_years
                t_r = 0.1075
                if not mkt_defaults['di_curve'].empty:
                    t_r = MarketDataService.interpolate_di_rate(def_exp, mkt_defaults['di_curve'])
                
                # Linha 1: Tempos e Proporção
                c1, c2, c3 = st.columns(3)
                t_vest = c1.number_input(f"Vesting (Anos) {i}", value=float(t.vesting_date), key=f"bn_v_{i}")
                t_life = c2.number_input(f"Vencimento (Anos) {i}", value=float(def_exp), key=f"bn_l_{i}")
                t_prop = c3.number_input(f"Peso % {i}", value=float(t.proportion*100), key=f"bn_p_{i}")/100
                
                # Linha 2: Mercado (Auto)
                c4, c5, c6 = st.columns(3)
                t_vol = c4.number_input(f"Volatilidade % {i}", value=float(mkt_defaults['vol']*100), key=f"bn_vol_{i}") / 100
                t_rate = c5.number_input(f"Taxa Risco % {i}", value=float(t_r*100), key=f"bn_r_{i}") / 100
                t_infl = c6.number_input(f"Corr. Strike % {i}", value=4.5 if analysis.has_strike_correction else 0.0, key=f"bn_inf_{i}") / 100
                
                # Linha 3: Comportamental
                c7, c8 = st.columns(2)
                t_lock = c7.number_input(f"Lockup (Anos) {i}", value=analysis.lockup_years, key=f"bn_lk_{i}")
                t_m = c8.number_input(f"Múltiplo M {i}", value=analysis.early_exercise_multiple, key=f"bn_m_{i}")

                inputs.append({
                    "vesting": t_vest, "T_life": t_life, "prop": t_prop,
                    "vol": t_vol, "r": t_rate, "infl": t_infl,
                    "lockup": t_lock, "m": t_m
                })

        if st.button("Calcular (Binomial)", type="primary"):
            bar = st.progress(0)
            total_fv = 0.0
            res = []
            for idx, inp in enumerate(inputs):
                fv = FinancialMath.binomial_custom_optimized(
                    S=S, K=K, r=inp["r"], vol=inp["vol"], q=q, 
                    vesting_years=inp["vesting"], 
                    turnover_w=analysis.turnover_rate,
                    multiple_M=inp["m"],
                    hurdle_H=0.0,
                    T_years=inp["T_life"],        
                    inflacao_anual=inp["infl"],
                    lockup_years=inp["lockup"]
                )
                w_fv = fv * inp["prop"]
                total_fv += w_fv
                res.append({
                    "Tranche": idx+1, 
                    "Vencimento": inp["T_life"],
                    "Taxa": f"{inp['r']:.2%}",
                    "Vol": f"{inp['vol']:.2%}",
                    "FV Unit": fv, 
                    "FV Ponderado": w_fv
                })
                bar.progress((idx+1)/len(inputs))
            
            st.metric("Resultado Binomial", f"R$ {total_fv:.4f}")
            st.dataframe(pd.DataFrame(res))

    def _render_rsu(self, S, q, analysis, mkt_defaults):
        st.info("ℹ️ Valuation de RSU / Phantom Shares")
        if mkt_defaults['auto']: st.success("🟢 Dados de Mercado Carregados.")
        
        self._manage_tranches()
        tranches = st.session_state['tranches']
        inputs = []

        for i, t in enumerate(tranches):
            with st.expander(f"Tranche {i+1}", expanded=True):
                # Interpolação DI para o prazo do Vesting (pagamento)
                t_r = 0.1075
                if not mkt_defaults['di_curve'].empty:
                    t_r = MarketDataService.interpolate_di_rate(t.vesting_date, mkt_defaults['di_curve'])

                c1, c2, c3 = st.columns(3)
                t_vest = c1.number_input(f"Vesting/Pagamento (Anos) {i}", value=float(t.vesting_date), key=f"rsu_v_{i}")
                t_prop = c2.number_input(f"Peso % {i}", value=float(t.proportion * 100), key=f"rsu_p_{i}") / 100
                t_rate = c3.number_input(f"Taxa Desconto % {i}", value=float(t_r*100), key=f"rsu_r_{i}") / 100
                
                c4, c5 = st.columns(2)
                t_lock = c4.number_input(f"Lock-up (Anos) {i}", value=float(analysis.lockup_years), key=f"rsu_l_{i}")
                
                # Volatilidade só é necessária para o desconto de Lockup (Chaffe)
                t_vol = 0.0
                if t_lock > 0:
                    t_vol = c5.number_input(f"Volatilidade % (Lockup) {i}", value=float(mkt_defaults['vol']*100), key=f"rsu_vol_{i}") / 100
                
                inputs.append({"T": t_vest, "prop": t_prop, "r": t_rate, "lockup": t_lock, "vol": t_vol})

        if st.button("Calcular Fair Value (RSU)"):
            total_fv = 0.0
            res_data = []
            
            for i, inp in enumerate(inputs):
                # Valor Presente do Ativo: S * exp(-q*T) * exp(-r*T) ??
                # NÃO. RSU não paga exercício (K=0).
                # O valor é S * exp(-q * T) se não paga dividendos na carência.
                # Não descontamos pelo risco livre (r) pois o ativo (S) já é valor presente.
                # A MENOS que seja liquidado em caixa no futuro, aí sim traz a valor presente?
                # IFRS 2 padrão: Fair Value na outorga é S_0. Desconta dividendos se não receber.
                
                base_fv = S * np.exp(-q * inp["T"])
                
                # Desconto de Lockup
                discount = 0.0
                if inp["lockup"] > 0:
                    discount = FinancialMath.calculate_lockup_discount(inp["vol"], inp["lockup"], base_fv, q)
                
                unit_fv = base_fv - discount
                weighted_fv = unit_fv * inp["prop"]
                total_fv += weighted_fv
                
                res_data.append({
                    "Tranche": i+1, "Vesting": inp["T"], 
                    "FV Unitário": unit_fv, "FV Ponderado": weighted_fv
                })
            
            st.metric("Fair Value Total (Ponderado)", f"R$ {total_fv:.4f}")
            st.dataframe(pd.DataFrame(res_data))

    def _render_monte_carlo_ai(self, S, K, q, analysis, text, api_key, mkt_defaults):
        st.warning("⚠️ Monte Carlo via Geração de Código IA")
        
        # Parâmetros sugeridos (usa média se auto)
        vol_sug = mkt_defaults['vol']
        r_sug = 0.1075
        if not mkt_defaults['di_curve'].empty:
            r_sug = MarketDataService.interpolate_di_rate(analysis.option_life_years, mkt_defaults['di_curve'])

        params = {
            "S0": S, "K": K, 
            "r": r_sug, 
            "sigma": vol_sug, 
            "q": q, 
            "T": analysis.option_life_years
        }
        
        c1, c2 = st.columns(2)
        if c1.button("1. Gerar Código"):
            with st.spinner("Gerando..."):
                code = DocumentService.generate_custom_monte_carlo_code(text, params, api_key)
                st.session_state['mc_code'] = code
        
        if st.session_state['mc_code']:
            code = st.text_area("Código Python", st.session_state['mc_code'], height=300)
            st.session_state['mc_code'] = code 
            
            if c2.button("2. Executar", type="primary"):
                old_stdout = io.StringIO()
                sys.stdout = old_stdout
                local_scope = {}
                try:
                    exec(code, local_scope)
                    output = old_stdout.getvalue()
                    sys.stdout = sys.__stdout__ 
                    
                    st.text(output)
                    if 'fv' in local_scope:
                        st.metric("Resultado Monte Carlo", f"R$ {local_scope['fv']:.4f}")
                except Exception as e:
                    sys.stdout = sys.__stdout__
                    st.error(f"Erro na execução: {e}")
