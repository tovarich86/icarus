"""
Módulo de Interface do Usuário (UI).

Responsável por renderizar os componentes visuais do Streamlit e orquestrar o fluxo
de interação do usuário, com foco em Remensuração e Conformidade IFRS 2.
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import sys
from typing import List, Dict

from core.domain import PlanAnalysisResult, Tranche, PricingModelType, SettlementType
from engines.financial import FinancialMath
from services.ai_service import DocumentService
from services.strategy import ModelSelectorService

class IFRS2App:
    """
    Aplicação principal do Icarus. Gerencia o estado da sessão e o layout.
    """

    def run(self) -> None:
        """Método principal de execução da interface."""
        st.title("🛡️ Icarus: Beta 1 (Modular)")
        
        # Inicialização de Estado (Session State)
        if 'analysis_result' not in st.session_state:
            st.session_state['analysis_result'] = None
        if 'full_context_text' not in st.session_state:
            st.session_state['full_context_text'] = ""
        if 'tranches' not in st.session_state:
            st.session_state['tranches'] = []
        if 'mc_code' not in st.session_state:
            st.session_state['mc_code'] = ""

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
            st.caption("v.Beta 2.0 - Foco Contábil")

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
        
        # Leitura de Arquivos
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
        
        # Chamada ao Serviço de IA
        if api_key:
            with st.spinner("🤖 IA Analisando estrutura do plano e classificação contábil..."):
                analysis = DocumentService.analyze_plan_with_gemini(combined_text, api_key)
        else:
            st.warning("⚠️ Sem API Key: Usando Mock.")
            analysis = DocumentService.mock_analysis(combined_text)
            
        if analysis:
            # Estratégia de Seleção de Modelo
            analysis = ModelSelectorService.select_model(analysis)
            st.session_state['analysis_result'] = analysis
            
            # Inicializa tranches editáveis
            if analysis.tranches:
                st.session_state['tranches'] = [t for t in analysis.tranches]
            else:
                st.session_state['tranches'] = [
                    Tranche(vesting_date=1.0, proportion=1.0, expiration_date=analysis.option_life_years)
                ]

    def _render_dashboard(self, analysis: PlanAnalysisResult, full_text: str, api_key: str) -> None:
        """Renderiza os resultados da análise e as calculadoras."""
        
        # --- Seção 1: Diagnóstico e Classificação Contábil ---
        st.subheader("1. Diagnóstico e Classificação Contábil")
        
        # Alerta de Liquidação (Passivo vs Equity)
        settlement = getattr(analysis, 'settlement_type', SettlementType.EQUITY_SETTLED)
        
        if settlement == SettlementType.CASH_SETTLED:
            st.error(f"⚠️ **CLASSIFICAÇÃO: PASSIVO (Liability)** - {settlement.value}")
            st.caption("Este instrumento é liquidado em caixa (ex: Phantom Shares, SARs). O IFRS 2 exige que o Fair Value seja **remensurado em toda data de balanço** até a liquidação.")
        elif settlement == SettlementType.HYBRID:
            st.warning(f"⚠️ **CLASSIFICAÇÃO: HÍBRIDO** - {settlement.value}. Verifique a política de liquidação provável.")
        else:
            st.success(f"✅ **CLASSIFICAÇÃO: EQUITY (Patrimônio)** - {settlement.value}")
            st.caption("Instrumento liquidado em ações. Mensurado na data de outorga (Grant Date). Não requer remensuração do FV, salvo modificações.")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### 📄 Resumo do Programa")
            prog_summary = getattr(analysis, 'program_summary', analysis.summary)
            st.info(prog_summary)
        with c2:
            st.markdown("##### 🧮 Parâmetros de Valuation")
            val_params = getattr(analysis, 'valuation_params', "Parâmetros não estruturados.")
            st.warning(val_params)

        st.divider()

        # --- Seção 2: Seleção de Metodologia ---
        st.subheader("2. Metodologia de Precificação")
        
        c_met1, c_met2 = st.columns([2, 1])
        with c_met1:
            st.markdown(f"**Modelo Recomendado:** `{analysis.model_recommended.value}`")
            st.write(analysis.methodology_rationale)
        with c_met2:
            st.caption("Justificativa Curta:")
            st.write(analysis.model_reason)

        # Seletor de Modelo Ativo
        opts = [m for m in PricingModelType if m != PricingModelType.UNDEFINED]
        try: idx = opts.index(analysis.model_recommended)
        except ValueError: idx = 0
        active_model = st.selectbox("Modelo Ativo (Cálculo):", opts, index=idx)
        
        st.divider()

        # --- Seção 3: Inputs de Mercado (Data Base) ---
        st.subheader("3. Parâmetros de Mercado (Data Base)")
        
        # Toggle para Contexto (Outorga vs Remensuração)
        calc_mode = st.radio("Contexto do Cálculo:", ["Data de Outorga (Grant)", "Remensuração (Reporting Date)"], horizontal=True)
        if calc_mode == "Remensuração (Reporting Date)" and settlement == SettlementType.EQUITY_SETTLED:
            st.warning("⚠️ Atenção: Instrumentos Equity-Settled geralmente não são remensurados, exceto em modificações contratuais.")

        col1, col2, col3, col4 = st.columns(4)
        S = col1.number_input("Preço da Ação (Spot) R$", 0.0, 10000.0, 50.0, help="Preço na data base (fechamento).")
        K = col2.number_input("Preço de Exercício (Strike) R$", 0.0, 10000.0, analysis.strike_price, help="Strike atualizado.")
        vol = col3.number_input("Volatilidade Anual (%)", 0.0, 500.0, 30.0, help="Volatilidade implícita ou histórica para o prazo remanescente.") / 100
        r = col4.number_input("Taxa Livre de Risco (%)", 0.0, 100.0, 10.75, help="Taxa spot (ex: DI Futuro / NTN-B) para o prazo remanescente.") / 100
        q = st.number_input("Dividend Yield Esperado (% a.a.)", 0.0, 100.0, 4.0) / 100

        st.subheader("4. Cálculo do Fair Value")

        # Roteamento
        if active_model == PricingModelType.BLACK_SCHOLES_GRADED:
            self._render_graded(S, K, r, vol, q, analysis, calc_mode)
        elif active_model == PricingModelType.BINOMIAL:
            self._render_binomial_graded(S, K, r, vol, q, analysis, calc_mode)
        elif active_model == PricingModelType.MONTE_CARLO:
            self._render_monte_carlo_ai(S, K, r, vol, q, analysis, full_text, api_key)
        elif active_model == PricingModelType.RSU:
            self._render_rsu(S, r, q, analysis, calc_mode)

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

    def _render_graded(self, S, K, r, vol, q, analysis, mode):
        st.info("ℹ️ Black-Scholes (Graded): Calcula cada tranche como uma opção independente.")
        self._manage_tranches()
        tranches = st.session_state['tranches']
        if not tranches: return

        inputs = []
        st.markdown("---")
        st.markdown(f"**Configuração das Tranches ({mode})**")
        st.caption("Nota: 'Prazo Vencimento' (T) é o input principal do BS. 'Vesting' é informativo para contabilidade.")

        for i, t in enumerate(tranches):
            with st.expander(f"Tranche {i+1}", expanded=True):
                c1, c2, c3 = st.columns(3)
                # Vesting Date (Carência)
                t_vest = c1.number_input(
                    f"Vesting (Anos)", 
                    value=float(t.vesting_date), 
                    min_value=0.0, step=0.1,
                    key=f"bs_v_{i}",
                    help="Tempo restante até a aquisição do direito."
                )
                
                # Expiration Date (Maturity / Expected Life)
                def_exp = t.expiration_date if t.expiration_date else analysis.option_life_years
                t_exp = c2.number_input(
                    f"Prazo Vencimento (T)", 
                    value=float(def_exp), 
                    min_value=0.01, step=0.1,
                    key=f"bs_t_{i}",
                    help="Tempo restante até o vencimento contratual ou vida esperada (Input do Modelo)."
                )
                
                t_prop = c3.number_input(f"Peso %", value=float(t.proportion*100), key=f"bs_p_{i}")/100
                inputs.append({"Vesting": t_vest, "T": t_exp, "prop": t_prop})

        if st.button("Calcular (Black-Scholes)", type="primary"):
            total_fv = 0.0
            res = []
            for idx, item in enumerate(inputs):
                # O Modelo BS usa o Tempo até Vencimento (T)
                fv = FinancialMath.bs_call(S, K, item["T"], r, vol, q)
                w_fv = fv * item["prop"]
                total_fv += w_fv
                res.append({
                    "Tranche": idx+1,
                    "Vesting (Anos)": item["Vesting"],
                    "Vencimento/T (Anos)": item["T"],
                    "FV Unitário": fv,
                    "FV Ponderado": w_fv
                })
            
            st.metric("Fair Value Total", f"R$ {total_fv:.4f}")
            st.dataframe(pd.DataFrame(res))

    def _render_binomial_graded(self, S, K, r, vol, q, analysis, mode):
        st.info("ℹ️ Modelo Lattice Binomial (Suporta Exercício Antecipado e Lock-up)")
        self._manage_tranches()
        tranches = st.session_state['tranches']
        inputs = []

        st.markdown(f"**Configuração das Tranches ({mode})**")
        
        for i, t in enumerate(tranches):
            with st.expander(f"Tranche {i+1}", expanded=False):
                c1, c2, c3 = st.columns(3)
                # Input explícito de Vesting vs Expiration
                t_vest = c1.number_input(f"Vesting (Anos) {i}", value=float(t.vesting_date), key=f"bn_v_{i}")
                
                def_exp = t.expiration_date if t.expiration_date else analysis.option_life_years
                t_life = c2.number_input(f"Vencimento (Anos) {i}", value=float(def_exp), key=f"bn_l_{i}")
                
                t_prop = c3.number_input(f"Peso % {i}", value=float(t.proportion*100), key=f"bn_p_{i}")/100
                
                c4, c5 = st.columns(2)
                t_lock = c4.number_input(f"Lockup (Anos) {i}", value=analysis.lockup_years, key=f"bn_lk_{i}")
                t_m = c5.number_input(f"Múltiplo M (Ex. Antecipado) {i}", value=analysis.early_exercise_multiple, key=f"bn_m_{i}")
                
                inputs.append({
                    "vesting": t_vest, "T_life": t_life, "prop": t_prop,
                    "lockup": t_lock, "m": t_m
                })

        if st.button("Calcular (Binomial)", type="primary"):
            bar = st.progress(0)
            total_fv = 0.0
            res = []
            for idx, inp in enumerate(inputs):
                fv = FinancialMath.binomial_custom_optimized(
                    S=S, K=K, r=r, vol=vol, q=q, 
                    vesting_years=inp["vesting"], # Define quando o exercício se torna possível
                    turnover_w=analysis.turnover_rate,
                    multiple_M=inp["m"],
                    hurdle_H=0.0,
                    T_years=inp["T_life"],        # Define o final da árvore
                    inflacao_anual=0.0, 
                    lockup_years=inp["lockup"]
                )
                w_fv = fv * inp["prop"]
                total_fv += w_fv
                res.append({
                    "Tranche": idx+1, 
                    "Vesting": inp["vesting"], 
                    "Vencimento": inp["T_life"],
                    "FV Unit": fv, 
                    "FV Ponderado": w_fv
                })
                bar.progress((idx+1)/len(inputs))
            
            st.metric("Resultado Binomial", f"R$ {total_fv:.4f}")
            st.dataframe(pd.DataFrame(res))

    def _render_rsu(self, S, r, q, analysis, mode):
        st.info("ℹ️ Valuation de RSU / Phantom Shares (Valor Intrínseco Descontado)")
        
        self._manage_tranches()
        tranches = st.session_state['tranches']
        tranche_inputs = []

        for i, t in enumerate(tranches):
            with st.expander(f"Tranche {i+1}", expanded=True):
                c1, c2, c3 = st.columns(3)
                # Para RSU, geralmente o pagamento é no Vesting, mas pode haver diferimento
                t_vest = c1.number_input(f"Vesting/Pagamento (Anos) {i}", value=float(t.vesting_date), key=f"rsu_v_{i}")
                
                t_lock = c2.number_input(f"Lock-up (Anos) {i}", value=float(analysis.lockup_years), key=f"rsu_l_{i}")
                t_prop = c3.number_input(f"Proporção % {i}", value=float(t.proportion * 100), key=f"rsu_prop_{i}") / 100
                
                # Volatilidade só é necessária se houver Lockup (Chaffe Model)
                t_vol = 0.30
                if t_lock > 0:
                    t_vol = st.number_input(f"Volatilidade % (Lockup) {i}", value=30.0, key=f"rsu_vol_{i}") / 100
                
                tranche_inputs.append({
                    "T": t_vest, "lockup": t_lock, "vol": t_vol, "prop": t_prop
                })

        st.divider()
        if st.button("Calcular Fair Value (RSU)"):
            total_fv = 0.0
            res_data = []
            
            for i, inp in enumerate(tranche_inputs):
                # Base Value: S * exp(-q * T)
                # Se não paga dividendos no vesting, desconta 'q'. Se paga, q=0 (ajuste no input global).
                base_fv = S * np.exp(-q * inp["T"])
                
                # Lockup Discount (Chaffe)
                discount = 0.0
                if inp["lockup"] > 0:
                    discount = FinancialMath.calculate_lockup_discount(inp["vol"], inp["lockup"], base_fv, q)
                
                unit_fv = base_fv - discount
                weighted_fv = unit_fv * inp["prop"]
                total_fv += weighted_fv
                
                res_data.append({
                    "Tranche": i+1, 
                    "Pagamento em": inp["T"], 
                    "FV Unitário": unit_fv,
                    "FV Ponderado": weighted_fv
                })
            
            st.metric("Fair Value Total (Ponderado)", f"R$ {total_fv:.4f}")
            st.dataframe(pd.DataFrame(res_data))

    def _render_monte_carlo_ai(self, S, K, r, vol, q, analysis, text, api_key):
        st.warning("⚠️ Monte Carlo via Geração de Código IA")
        
        # Usa option_life_years como padrão para T
        params = {"S0": S, "K": K, "r": r, "sigma": vol, "q": q, "T": analysis.option_life_years}
        
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
