import streamlit as st
import pandas as pd
import numpy as np
import io
import sys
from datetime import date, timedelta
from typing import List, Dict
from services.report_service import ReportService

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
        
        # --- NOVO: Estrutura de Abas ---
        tab_calc, tab_laudo = st.tabs(["🧮 1. Cálculo & Valuation", "📝 2. Gerador de Laudo"])
        
        with tab_calc:
            # Aqui vai todo o conteúdo original que estava no run()
            # Copie e cole a lógica de Header, Sidebar, if session_state...
            self._render_main_valuation_interface() # (Exemplo: refatore o conteúdo antigo para um método auxiliar ou mantenha aqui)

        with tab_laudo:
            self._render_report_interface()

    # Crie este novo método para desenhar a tela de preenchimento
    def _render_report_interface(self):
        st.header("Gerador de Laudo Contábil (CPC 10)")
        
        # Validação: Só permite gerar laudo se houver cálculo realizado
        if not st.session_state.get('last_calc_results'):
            st.warning("⚠️ Nenhum cálculo encontrado. Por favor, realize o valuation na aba 'Cálculo & Valuation' primeiro.")
            return

        if ReportService is None:
            st.error("Erro Crítico: O serviço 'ReportService' não foi carregado.")
            return

        # --- SEÇÃO 1: INPUTS DO USUÁRIO ---
        with st.container(border=True):
            st.subheader("1. Dados da Empresa e Programa")
            c1, c2 = st.columns(2)
            with c1:
                emp_nome = st.text_input("Razão Social", "Minha Empresa S.A.", key="rep_emp_nome")
                emp_ticker = st.text_input("Ticker (opcional)", "TICK3", key="rep_emp_ticker")
                emp_aberta = st.checkbox("Capital Aberto?", value=True, key="rep_emp_aberta")
            with c2:
                prog_nome = st.text_input("Nome do Plano", "Plano de Opção de Compra 2024", key="rep_prog_nome")
                prog_data = st.date_input("Data de Outorga", date.today(), key="rep_prog_data")
                prog_qtd = st.number_input("Qtd. Beneficiários", 1, 10000, 10, key="rep_prog_qtd")

        with st.container(border=True):
            st.subheader("2. Premissas Contábeis e Responsável")
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Contabilidade**")
                turnover_contab = st.number_input("Turnover Esperado (% a.a.)", 0.0, 50.0, 5.0, key="rep_turnover") / 100
                tem_encargos = st.checkbox("Incide Encargos Sociais (INSS)?", False, key="rep_encargos")
                perf_nao_mercado = st.checkbox("Possui Metas Não-Mercado (EBITDA)?", False, key="rep_metas")
                perc_atingimento = 100.0
                if perf_nao_mercado:
                    perc_atingimento = st.number_input("% Atingimento Esperado", 0.0, 200.0, 100.0, key="rep_ating")
            with c2:
                st.markdown("**Responsável Técnico**")
                resp_nome = st.text_input("Nome", "Consultor Responsável", key="rep_resp_nome")
                resp_cargo = st.text_input("Cargo", "Especialista em Remuneração", key="rep_resp_cargo")
                resp_email = st.text_input("Email", "contato@exemplo.com", key="rep_resp_email")

        # --- SEÇÃO 2: GERAÇÃO AUTOMÁTICA (TEMPLATE HOSPEDADO) ---
        st.subheader("3. Geração do Documento")
        
        # Lógica para encontrar o template hospedado
        import os
        possible_paths = [
            "TEMPLATE_FINAL_PADRAO.docx",             # Raiz
            "templates/TEMPLATE_FINAL_PADRAO.docx",   # Pasta templates
            "ui/TEMPLATE_FINAL_PADRAO.docx"           # Pasta ui
        ]
        
        template_path = None
        for p in possible_paths:
            if os.path.exists(p):
                template_path = p
                break
        
        if not template_path:
            st.error("❌ ERRO DE SISTEMA: O arquivo de template padrão ('TEMPLATE_FINAL_PADRAO.docx') não foi encontrado no servidor. Contate o administrador do repositório.")
            return

        st.info(f"✅ Template padrão carregado com sucesso.")

        if st.button("📄 Gerar Laudo Oficial", type="primary"):
            # Consolida todos os inputs manuais
            manual_inputs = {
                "empresa": {"nome": emp_nome, "ticker": emp_ticker, "capital_aberto": emp_aberta},
                "programa": {"nome": prog_nome, "data_outorga": prog_data, "qtd_beneficiarios": prog_qtd},
                "responsavel": {"nome": resp_nome, "cargo": resp_cargo, "email": resp_email},
                "contab": {
                    "taxa_turnover": turnover_contab,
                    "tem_metas_nao_mercado": perf_nao_mercado,
                    "percentual_atingimento": perc_atingimento,
                    "tem_encargos": tem_encargos
                }
            }

            try:
                with st.spinner("Compilando dados e gerando documento..."):
                    # Chama o serviço para mapear os dados
                    context = ReportService.generate_report_context(
                        st.session_state['analysis_result'],
                        st.session_state['tranches'],
                        st.session_state['last_calc_results'],
                        manual_inputs
                    )
                    
                    # Gera o binário do arquivo usando o PATH encontrado
                    docx_bytes = ReportService.render_template(template_path, context)
                
                # Botão de Download
                st.download_button(
                    label=f"💾 Baixar Laudo: {emp_nome}.docx",
                    data=docx_bytes,
                    file_name=f"Laudo_{emp_nome.replace(' ', '_')}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
                st.success("Documento gerado! Clique no botão acima para salvar.")
                
            except Exception as e:
                st.error(f"Erro ao processar o documento: {str(e)}")

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
            st.caption("v.Release 1.7 - UX/UI Boost")

        # --- Crie as abas ---
        tab_calc, tab_laudo = st.tabs(["🧮 1. Cálculo & Valuation", "📝 2. Gerador de Laudo"])

        # --- Aba 1: Mantém sua lógica atual ---
        with tab_calc:
            if st.session_state['analysis_result']:
                self._render_dashboard(
                    st.session_state['analysis_result'], 
                    st.session_state['full_context_text'], 
                    gemini_key
                )
            else:
                self._render_empty_state()

        # --- Aba 2: Nova funcionalidade ---
        with tab_laudo:
            self._render_report_interface()

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
        
        # Container Visual
        with st.container():
            c1, c2 = st.columns([1, 1]) # Divisão 50% / 50%
            
            # --- COLUNA DA ESQUERDA: Status + Resumo do Plano ---
            with c1:
                settlement = getattr(analysis, 'settlement_type', SettlementType.EQUITY_SETTLED)
                model_label = analysis.model_recommended.value if analysis.model_recommended else "Indefinido"

                # 1. Caixa de Status (Verde/Vermelho)
                if settlement == SettlementType.CASH_SETTLED:
                    st.error(f"⚠️ PASSIVO (Liability) - {settlement.value}\n\n **Modelo Recomendado:** {model_label}")
                else:
                    st.success(f"✅ EQUITY (Patrimônio) - {settlement.value}\n\n **Modelo Recomendado:** {model_label}")
                
                # 2. Caixa Azul: Agora exibe o RESUMO DO PLANO (Features)
                # Usamos program_summary (resumo narrativo) ou contract_features (lista de cláusulas)
                resumo_txt = analysis.program_summary if analysis.program_summary else analysis.contract_features
                st.info(f"📋 **Resumo do Programa:**\n\n{resumo_txt}")

            # --- COLUNA DA DIREITA: Premissas de Valuation ---
            with c2:
                st.markdown("### 📊 Premissas de Valuation")
                
                # Usamos st.markdown para garantir que as quebras de linha e bullet points funcionem
                # O valuation_params virá formatado da IA
                params_txt = getattr(analysis, 'valuation_params', analysis.summary)
                st.markdown(params_txt)

        st.divider()

        st.subheader("3. Premissas de Mercado")
        c1, c2, c3, c4 = st.columns(4)
        
        # Tooltips adicionados para clareza
        S = c1.number_input("Spot (R$)", 0.0, 10000.0, 50.0, help="Preço atual da ação-objeto (Data Base).")
        K = c2.number_input("Strike (R$)", 0.0, 10000.0, analysis.strike_price, help="Preço de exercício da opção.")
        q = c3.number_input("Div Yield (%)", 0.0, 100.0, 0.0, help="Expectativa de dividendos anuais (Dividend Yield).") / 100
        
        opts = [m for m in PricingModelType if m != PricingModelType.UNDEFINED]
        idx = opts.index(analysis.model_recommended) if analysis.model_recommended in opts else 0
        
        # Correção do Seletor: Mostra apenas o nome amigável (sem 'PricingModelType.')
        active_model = c4.selectbox(
            "Modelo de Precificação", 
            opts, 
            index=idx,
            format_func=lambda x: x.value 
        )

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
                # Cabeçalho Limpo
                st.markdown(f"##### 🔹 Tranche {i+1}")
                
                # --- LINHA 1: Tempo e Peso (Grid 4 colunas) ---
                c_time1, c_time2, c_time3, c_time4 = st.columns(4)
                
                with c_time1:
                    def_exp = t.expiration_date if t.expiration_date else analysis.option_life_years
                    t_exp = st.number_input(
                        f"Vencimento (Anos)", 
                        value=float(def_exp), 
                        key=f"bi_t_{i}",
                        help="Prazo contratual total da opção (Life)."
                    )
                with c_time2:
                    t_vest = st.number_input(
                        f"Vesting (Anos)", 
                        value=float(t.vesting_date), 
                        key=f"bi_v_{i}",
                        help="Período de carência até o direito se tornar exercível."
                    )
                with c_time3:
                    t_prop = st.number_input(
                        f"Peso (%)", 
                        value=float(t.proportion*100), 
                        key=f"bi_p_{i}",
                        help="% do total de opções que pertence a esta tranche."
                    )/100
                with c_time4:
                    t_lock = st.number_input(
                        f"Lockup (Anos)", 
                        value=float(analysis.lockup_years), 
                        key=f"bi_lock_{i}",
                        help="Tempo de restrição de venda da ação após o exercício."
                    )

                # --- LINHA 2: Mercado (Vol e Rate lado a lado) ---
                c_mkt1, c_mkt2 = st.columns(2)
                with c_mkt1:
                    self._render_vol_widget(i, "bi")
                with c_mkt2:
                    self._render_rate_widget_table(i, "bi", t_exp)

                # --- LINHA 3: Parâmetros Avançados (Expander) ---
                with st.expander("⚙️ Parâmetros Avançados (Turnover & Barreiras)", expanded=False):
                    c_adv1, c_adv2, c_adv3 = st.columns(3)
                    
                    t_turnover = c_adv1.number_input(
                        f"Turnover Anual (%)", 
                        value=float(analysis.turnover_rate * 100), 
                        key=f"bi_turn_{i}",
                        help="Taxa estimada de saída de funcionários antes do vesting."
                    ) / 100
                    
                    t_m = c_adv2.number_input(
                        f"Múltiplo M", 
                        value=float(analysis.early_exercise_multiple), 
                        key=f"bi_m_{i}",
                        help="Gatilho de exercício antecipado (Ex: 2.0x o Strike)."
                    )
                    
                    t_strike_corr = c_adv3.number_input(
                        f"Corr. Strike (% a.a.)", 
                        value=4.5 if analysis.has_strike_correction else 0.0, 
                        key=f"bi_corr_{i}",
                        help="Taxa de correção monetária do Strike (ex: IGPM)."
                    ) / 100

                key_vol = f"vol_val_bi_{i}"
                key_rate = f"rate_val_bi_{i}"
                
                inputs.append({
                    "TrancheID": i+1, "S": S, "K": K, "q": q,
                    "T": t_exp, "Vesting": t_vest, "Prop": t_prop,
                    "Vol": st.session_state.get(key_vol, 0.30),
                    "r": st.session_state.get(key_rate, 0.001075),
                    "Turnover": t_turnover, "M": t_m, 
                    "StrikeCorr": t_strike_corr, "Lockup": t_lock
                })

        if st.button("Calcular (Binomial)", type="primary", use_container_width=True):
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
        st.info("ℹ️ Black-Scholes (Graded): Cálculo padrão para opções sem barreiras complexas.")
        self._manage_tranches_buttons()
        tranches = st.session_state['tranches']
        inputs = []
        
        for i, t in enumerate(tranches):
            with st.container(border=True):
                st.markdown(f"##### 🔹 Tranche {i+1}")
                
                # Linha 1: Dados da Opção
                c1, c2, c3 = st.columns(3)
                with c1:
                    def_exp = t.expiration_date if t.expiration_date else analysis.option_life_years
                    t_exp = st.number_input("Vencimento (Anos)", value=float(def_exp), key=f"bs_t_{i}", help="Prazo total (Life).")
                with c2:
                    t_prop = st.number_input("Peso (%)", value=float(t.proportion*100), key=f"bs_p_{i}", help="Peso desta tranche no total.")/100
                with c3:
                    t_vest = st.number_input("Vesting (Ref)", value=float(t.vesting_date), key=f"bs_v_{i}", help="Apenas informativo no BS.")
                
                # Linha 2: Mercado
                c_mkt1, c_mkt2 = st.columns(2)
                with c_mkt1:
                    self._render_vol_widget(i, "bs")
                with c_mkt2:
                    self._render_rate_widget_table(i, "bs", t_exp)

                key_vol = f"vol_val_bs_{i}"
                key_rate = f"rate_val_bs_{i}"

                inputs.append({
                    "TrancheID": i+1, "S": S, "K": K, "q": q,
                    "T": t_exp, "Prop": t_prop,
                    "Vol": st.session_state.get(key_vol, 0.30),
                    "r": st.session_state.get(key_rate, 0.001075)
                })

        if st.button("Calcular (Black-Scholes)", type="primary", use_container_width=True):
            self._execute_calc(inputs, PricingModelType.BLACK_SCHOLES_GRADED)

    def _render_rsu(self, S, q, analysis):
        self._manage_tranches_buttons()
        tranches = st.session_state['tranches']
        inputs = []
        
        for i, t in enumerate(tranches):
            with st.container(border=True):
                st.markdown(f"##### 🔹 Tranche {i+1}")
                c1, c2, c3 = st.columns(3)
                t_vest = c1.number_input(f"Pagamento (Anos)", value=float(t.vesting_date), key=f"rsu_v_{i}", help="Data do recebimento da ação.")
                t_prop = c2.number_input(f"Peso (%)", value=float(t.proportion*100), key=f"rsu_p_{i}", help="Proporção do total.")/100
                t_lock = c3.number_input(f"Lockup (Anos)", value=float(analysis.lockup_years), key=f"rsu_l_{i}", help="Restrição de venda pós-vesting.")
                
                vol_val = 0.30
                if t_lock > 0:
                    st.caption("Volatilidade para Lockup:")
                    self._render_vol_widget(i, "rsu")
                    vol_val = st.session_state.get(f"vol_val_rsu_{i}", 0.30)
                
                inputs.append({
                    "TrancheID": i+1, "S": S, "q": q, "T": t_vest, 
                    "Prop": t_prop, "Lockup": t_lock, "Vol": vol_val
                })
        
        if st.button("Calcular (RSU)", type="primary", use_container_width=True):
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
                    opts = {
                        f"EWMA: {summ['mean_ewma']*100:.1f}%": summ['mean_ewma']*100,
                        f"Hist: {summ['mean_std']*100:.1f}%": summ['mean_std']*100
                    }
                    if summ.get('mean_garch', 0) > 0:
                        opts[f"GARCH: {summ['mean_garch']*100:.1f}%"] = summ['mean_garch']*100
                    
                    sel = st.radio("Métrica", list(opts.keys()), key=f"rad_{prefix}_{i}")
                    
                    st.button("Aplicar", key=f"app_{prefix}_{i}", 
                              on_click=self._update_widget_state, args=(key_val, key_w, opts[sel]))
                              
                    # --- NOVO: Botão de Download para Auditoria ---
                    if "audit_excel" in res and res["audit_excel"]:
                         st.download_button(
                            label="💾 Baixar Auditoria (Excel)",
                            data=res["audit_excel"],
                            file_name=f"auditoria_volatilidade_tranche_{i+1}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key=f"dl_vol_{prefix}_{i}"
                        )

    def _render_rate_widget_table(self, i, prefix, t_years):
        """
        WIDGET DE DI OTIMIZADO: Callbacks Seguros + Tabela Limpa.
        """
        st.markdown("Taxa DI (%)")
        c_in, c_pop = st.columns([0.85, 0.15])
        
        key_val = f"rate_val_{prefix}_{i}"
        key_w = f"rate_w_{prefix}_{i}"
        
        if key_val not in st.session_state: 
            st.session_state[key_val] = .1075
        
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
                
                # --- CORREÇÃO DO ERRO DE WIDTH AQUI ---
                st.dataframe(
                    df_show[[col_venc, 'Taxa (%)']].rename(columns={col_venc: 'Vencimento'}), 
                    use_container_width=True, # Corrigido de width='stretch'
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
                
                # CALLBACK SEGURO
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
                # Inputs Básicos
                raw_vol = item.get('Vol', 0.0)
                vol_decimal = float(raw_vol) / 100.0 
                
                S = float(item['S'])
                T = float(item['T'])
                q = float(item['q'])
                
                # Get Seguro para Inputs Opcionais
                K = float(item.get('K', 0.0))
                r = float(item.get('r', 0.0))
                
                vesting = float(item.get('Vesting', 0.0))
                turnover = float(item.get('Turnover', 0.0))
                
                # Trava de Segurança M
                raw_m = float(item.get('M', 2.0))
                if raw_m < 1.0:
                    mul_m = 1.0
                else:
                    mul_m = raw_m

                strike_corr = float(item.get('StrikeCorr', 0.0)) 
                lockup = float(item.get('Lockup', 0.0))

                # Chamada aos Motores
                fv = 0.0
                if model_type == PricingModelType.BINOMIAL:
                    fv = FinancialMath.binomial_custom_optimized(
                        S, K, r, vol_decimal, q,
                        vesting, turnover, mul_m, 0.0,
                        T, strike_corr, lockup
                    )
                elif model_type == PricingModelType.BLACK_SCHOLES_GRADED:
                    fv = FinancialMath.bs_call(S, K, T, r, vol_decimal, q)
                elif model_type == PricingModelType.RSU:
                    base = S * np.exp(-q * T)
                    disc = 0.0
                    if lockup > 0:
                        disc = FinancialMath.calculate_lockup_discount(vol_decimal, lockup, base, q)
                    fv = base - disc
                
                w_fv = fv * float(item['Prop'])
                total_fv += w_fv
                
                res_data.append({
                    "Tranche": item['TrancheID'],
                    "FV Unit": fv,
                    "FV Ponderado": w_fv,
                    "M Ajustado": mul_m 
                })
            except Exception as e:
                st.error(f"Erro Tranche {idx+1}: {e}")
                
            prog.progress((idx+1)/len(inputs))

        st.session_state['last_calc_results'] = res_data
        st.toast("Cálculo salvo! Acesse a aba 'Gerador de Laudo'.", icon="💾")
        
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
