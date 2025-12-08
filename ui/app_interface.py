"""
Módulo de Interface do Usuário (UI).

Responsável por renderizar os componentes visuais do Streamlit (botões, gráficos, inputs)
e orquestrar o fluxo de interação do usuário.
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import sys
from typing import List, Dict

from core.domain import PlanAnalysisResult, Tranche, PricingModelType
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
            
            # Gestão de API Key (Prioriza Secrets > Input Manual)
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
            with st.spinner("🤖 IA Analisando estrutura do plano..."):
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
                st.session_state['tranches'] = [Tranche(1.0, 1.0)]

    def _render_dashboard(self, analysis: PlanAnalysisResult, full_text: str, api_key: str) -> None:
        """Renderiza os resultados da análise e as calculadoras."""
        
        # --- Seção 1: Diagnóstico e Estrutura ---
        st.subheader("1. Diagnóstico e Estrutura")
        
        # Layout em Colunas para separar Resumo (Jurídico) x Parâmetros (Quant)
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("##### 📄 Resumo do Programa")
            # Tenta pegar os novos campos, com fallback seguro para versões antigas do objeto
            prog_summary = getattr(analysis, 'program_summary', analysis.summary)
            st.info(prog_summary)

        with c2:
            st.markdown("##### 🧮 Parâmetros de Valuation")
            val_params = getattr(analysis, 'valuation_params', "Parâmetros não estruturados.")
            st.warning(val_params)

        st.divider()

        # --- Seção 2: Seleção de Metodologia ---
        st.subheader("2. Seleção de Metodologia")
        
        # Destaque para o modelo recomendado
        st.markdown(f"**Modelo Recomendado:** `{analysis.model_recommended.value}`")
        st.caption(analysis.methodology_rationale)

        # Expander com detalhes comparativos
        with st.expander("Ver Análise Comparativa (Prós/Contras)"):
            st.write(analysis.model_comparison)
            cp, cc = st.columns(2)
            cp.write("**Prós:**")
            for p in analysis.pros: cp.write(f"- {p}")
            cc.write("**Contras:**")
            for c in analysis.cons: cc.write(f"- {c}")

        st.divider()

        # Seletor de Modelo Ativo
        opts = [m for m in PricingModelType if m != PricingModelType.UNDEFINED]
        try: idx = opts.index(analysis.model_recommended)
        except ValueError: idx = 0
        active_model = st.selectbox("Modelo Ativo (Cálculo):", opts, index=idx)
        
        st.divider()

        # --- Seção 3: Inputs de Mercado ---
        st.subheader("3. Parâmetros de Mercado (Base)")
        col1, col2, col3, col4 = st.columns(4)
        S = col1.number_input("Preço Spot (R$)", 0.0, 10000.0, 50.0)
        K = col2.number_input("Strike (R$)", 0.0, 10000.0, analysis.strike_price)
        vol = col3.number_input("Volatilidade (%)", 0.0, 500.0, 30.0) / 100
        r = col4.number_input("Taxa Livre Risco (%)", 0.0, 100.0, 10.75) / 100
        q = st.number_input("Dividend Yield (% a.a.)", 0.0, 100.0, 4.0) / 100

        st.subheader("4. Cálculo do Fair Value")

        # Roteamento para renderizadores específicos
        if active_model == PricingModelType.BLACK_SCHOLES_GRADED:
            self._render_graded(S, K, r, vol, q, analysis)
        elif active_model == PricingModelType.BINOMIAL:
            self._render_binomial_graded(S, K, r, vol, q, analysis)
        elif active_model == PricingModelType.MONTE_CARLO:
            self._render_monte_carlo_ai(S, K, r, vol, q, analysis, full_text, api_key)
        elif active_model == PricingModelType.RSU:
            self._render_rsu(S, r, q, analysis)

    def _manage_tranches(self) -> None:
        """Widget auxiliar para adicionar/remover tranches."""
        st.markdown("#### ⚙️ Gerenciar Tranches")
        c1, c2 = st.columns(2)
        if c1.button("➕ Adicionar Tranche"):
            last_vesting = st.session_state['tranches'][-1].vesting_date if st.session_state['tranches'] else 0.0
            st.session_state['tranches'].append(Tranche(last_vesting + 1.0, 0.0))
            st.rerun()
        if c2.button("➖ Remover Última"):
            if len(st.session_state['tranches']) > 0:
                st.session_state['tranches'].pop()
                st.rerun()

    def _render_graded(self, S, K, r, vol, q, analysis):
        self._manage_tranches()
        tranches = st.session_state['tranches']
        if not tranches: return

        inputs = []
        for i, t in enumerate(tranches):
            with st.expander(f"Tranche {i+1} ({t.vesting_date} anos)", expanded=True):
                c1, c2 = st.columns(2)
                t_vest = c1.number_input(f"Vesting {i}", value=float(t.vesting_date), key=f"bs_t_{i}")
                t_prop = c2.number_input(f"Peso % {i}", value=float(t.proportion*100), key=f"bs_p_{i}")/100
                inputs.append({"T": t_vest, "prop": t_prop})

        if st.button("Calcular (Black-Scholes)", type="primary"):
            total_fv = 0.0
            res = []
            for item in inputs:
                fv = FinancialMath.bs_call(S, K, item["T"], r, vol, q)
                w_fv = fv * item["prop"]
                total_fv += w_fv
                res.append({"Vesting": item["T"], "Unit FV": fv, "Weighted FV": w_fv})
            
            st.metric("Fair Value Total", f"R$ {total_fv:.4f}")
            st.dataframe(pd.DataFrame(res))

    def _render_binomial_graded(self, S, K, r, vol, q, analysis):
        st.info("ℹ️ Modelo Lattice Binomial (Suporta Exercício Antecipado e Lock-up)")
        self._manage_tranches()
        tranches = st.session_state['tranches']
        inputs = []

        for i, t in enumerate(tranches):
            with st.expander(f"Tranche {i+1}", expanded=False):
                # Inputs simplificados para brevidade, mas expansíveis
                c1, c2, c3 = st.columns(3)
                t_life = c1.number_input(f"Vida Total {i}", value=analysis.option_life_years, key=f"bn_l_{i}")
                t_lock = c2.number_input(f"Lockup {i}", value=analysis.lockup_years, key=f"bn_lk_{i}")
                t_m = c3.number_input(f"Múltiplo M {i}", value=analysis.early_exercise_multiple, key=f"bn_m_{i}")
                
                inputs.append({
                    "vesting": t.vesting_date, "prop": t.proportion,
                    "T_life": t_life, "lockup": t_lock, "m": t_m
                })

        if st.button("Calcular (Binomial)", type="primary"):
            bar = st.progress(0)
            total_fv = 0.0
            res = []
            for idx, inp in enumerate(inputs):
                fv = FinancialMath.binomial_custom_optimized(
                    S=S, K=K, r=r, vol=vol, q=q, 
                    vesting_years=inp["vesting"],
                    turnover_w=analysis.turnover_rate,
                    multiple_M=inp["m"],
                    hurdle_H=0.0,
                    T_years=inp["T_life"],
                    inflacao_anual=0.0, 
                    lockup_years=inp["lockup"]
                )
                w_fv = fv * inp["prop"]
                total_fv += w_fv
                res.append({"Vesting": inp["vesting"], "FV Unit": fv, "FV Ponderado": w_fv})
                bar.progress((idx+1)/len(inputs))
            
            st.metric("Resultado Binomial", f"R$ {total_fv:.4f}")
            st.dataframe(pd.DataFrame(res))

    def _render_rsu(self, S, r, q, analysis):
        st.info("ℹ️ Valuation de RSU / Matching Shares (Por Tranche)")
        st.caption("Cálculo: Preço da Ação descontado de Dividendos e Lock-up.")
        
        self._manage_tranches()
        tranches = st.session_state['tranches']
        tranche_inputs = []

        if not tranches:
            st.warning("Nenhuma tranche definida.")
            return

        for i, t in enumerate(tranches):
            with st.expander(f"Tranche {i+1}", expanded=True):
                c1, c2, c3, c4 = st.columns(4)
                t_vest = c1.number_input(f"Vesting {i}", value=float(t.vesting_date), key=f"rsu_v_{i}")
                t_lock = c2.number_input(f"Lock-up {i}", value=float(analysis.lockup_years), key=f"rsu_l_{i}")
                t_vol = c3.number_input(f"Volatilidade % (Lockup) {i}", value=30.0, key=f"rsu_vol_{i}") / 100
                t_prop = c4.number_input(f"Proporção % {i}", value=float(t.proportion * 100), key=f"rsu_prop_{i}") / 100
                
                tranche_inputs.append({
                    "T": t_vest, "lockup": t_lock, "vol": t_vol, "prop": t_prop
                })

        st.divider()
        if st.button("Calcular Fair Value (RSU)"):
            total_fv = 0.0
            res_data = []
            
            for i, inp in enumerate(tranche_inputs):
                # Base Value: S * exp(-q * T)
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
                    "Vesting": inp["T"], 
                    "FV Unitário": unit_fv,
                    "FV Ponderado": weighted_fv
                })
            
            st.metric("Fair Value Total (Ponderado)", f"R$ {total_fv:.4f}")
            st.dataframe(pd.DataFrame(res_data))

    def _render_monte_carlo_ai(self, S, K, r, vol, q, analysis, text, api_key):
        st.warning("⚠️ Monte Carlo via Geração de Código IA")
        
        params = {"S0": S, "K": K, "r": r, "sigma": vol, "T": analysis.option_life_years}
        
        c1, c2 = st.columns(2)
        if c1.button("1. Gerar Código"):
            with st.spinner("Gerando..."):
                code = DocumentService.generate_custom_monte_carlo_code(text, params, api_key)
                st.session_state['mc_code'] = code
        
        if st.session_state['mc_code']:
            code = st.text_area("Código Python", st.session_state['mc_code'], height=300)
            st.session_state['mc_code'] = code # Save edits
            
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
