import streamlit as st
import os
from datetime import date
from ui.state import AppState
from services.report_service import ReportService
from core.domain import PricingModelType, SettlementType

def render_report_tab():
    results = AppState.get_calc_results()
    analysis = AppState.get_analysis()
    
    if not results or not analysis:
        st.warning("⚠️ Nenhum cálculo disponível. Realize o valuation na aba anterior primeiro.")
        return

    st.header("Gerador de Laudo Contábil (CPC 10)")

    # --- 1. DADOS DA EMPRESA E PROGRAMA ---
    with st.container(border=True):
        st.subheader("1. Dados da Empresa e Programa")
        c1, c2 = st.columns(2)
        with c1:
            emp_nome = st.text_input("Razão Social", "Minha Empresa S.A.", key="rep_emp_nome")
            emp_ticker = st.text_input("Ticker", "TICK3", key="rep_emp_ticker")
            emp_aberta = st.checkbox("Capital Aberto?", value=True, key="rep_emp_aberta")
            
            if emp_aberta:
                nome_bolsa = st.text_input("Bolsa", "B3 S.A. - Brasil, Bolsa, Balcão", key="rep_bolsa")
                metodo_privado = ""
            else:
                nome_bolsa = ""
                metodo_privado = st.text_input("Metodologia (Privada)", "Fluxo de Caixa Descontado", key="rep_metodo_priv")

        with c2:
            prog_nome = st.text_input("Nome do Plano", "Plano de Stock Options 2025", key="rep_prog_nome")
            
            tipo_detalhado = st.selectbox("Tipo do Plano (Texto)", 
                                        ["Plano de Opção de Compra de Ações (Stock Options)", 
                                         "Plano de Ações Restritas (Restricted Shares)",
                                         "Plano de Ações por Performance (Performance Shares)",
                                         "Plano de Phantom Options"],
                                        key="rep_tipo_detalhado")
            
            prog_data = st.date_input("Data de Outorga", date.today(), key="rep_prog_data")
            prog_qtd = st.number_input("Qtd. Beneficiários", 1, 10000, 10, key="rep_prog_qtd")

    # --- 2. PREMISSAS TÉCNICAS E CONTÁBEIS (RESTAURADO) ---
    with st.container(border=True):
        st.subheader("2. Premissas Técnicas e Contábeis")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Parâmetros de Mercado**")
            
            moeda_selecionada = st.selectbox(
                "Curva de Juros (Moeda)", 
                ["BRL (DI1 - Brasil)", "USD (Treasury Bond - EUA)", "EUR (Euro)"],
                key="rep_moeda"
            )
            
            # Tenta inferir o cenário atual
            div_yield_calc = results[0].get('q', 0.0) if results else 0.0
            idx_div = 0 if div_yield_calc == 0 else 2 
            
            cenario_div = st.selectbox(
                "Cenário de Dividendos",
                ["ZERO (Sem expectativa)", "PAGO (Protegido/Reinvestido)", "PENALIZA (Desconto no FV)"],
                index=idx_div,
                key="rep_cenario_div"
            )

            tem_correcao = analysis.has_strike_correction
            indice_texto = ""
            if tem_correcao:
                indice_texto = st.text_input("Índice de Correção", "IGPM", key="rep_indice")

        with c2:
            st.markdown("**Parâmetros Contábeis**")
            turnover_contab = st.number_input("Turnover Esperado (% a.a.)", 0.0, 50.0, 5.0, key="rep_turnover") / 100
            tem_encargos = st.checkbox("Incide Encargos Sociais (INSS)?", False, key="rep_encargos")
            
            st.markdown("---")
            st.markdown("**Performance (Não-Mercado)**")
            tem_nao_mercado = st.checkbox(
                "Possui Metas Internas (KPIs)?", 
                value=False, 
                key="rep_flag_kpi",
                help="EBITDA, Lucro Líquido, etc."
            )
            
            perc_atingimento = 1.0 
            if tem_nao_mercado:
                perc_atingimento = st.number_input(
                    "Exp. Atingimento (%)", 
                    value=100.0, 
                    key="rep_perc_kpi"
                ) / 100.0

            st.markdown("---")
            st.markdown("**Responsável Técnico**")
            resp_nome = st.text_input("Nome", "Consultor Responsável", key="rep_resp_nome")
            resp_cargo = st.text_input("Cargo", "Especialista", key="rep_resp_cargo")
            resp_email = st.text_input("Email", "contato@exemplo.com", key="rep_resp_email")

    # --- 3. GERAÇÃO ---
    st.subheader("3. Geração do Documento")
    
    if st.button("📄 Gerar Laudo Oficial", type="primary"):
        _generate_docx_full(
            emp_nome, emp_ticker, emp_aberta, nome_bolsa, metodo_privado,
            prog_nome, prog_data, prog_qtd, tipo_detalhado,
            moeda_selecionada, cenario_div, indice_texto,
            turnover_contab, tem_encargos, tem_nao_mercado, perc_atingimento,
            resp_nome, resp_cargo, resp_email,
            analysis, results
        )

def _generate_docx_full(
    emp_nome, emp_ticker, emp_aberta, nome_bolsa, metodo_privado,
    prog_nome, prog_data, prog_qtd, tipo_detalhado,
    moeda_selecionada, cenario_div, indice_texto,
    turnover_contab, tem_encargos, tem_nao_mercado, perc_atingimento,
    resp_nome, resp_cargo, resp_email,
    analysis, results
):
    try:
        # Lógica de Modelo
        modelo_atual = analysis.model_recommended
        metodologia_str = "BLACK_SCHOLES"
        if modelo_atual == PricingModelType.BINOMIAL: metodologia_str = "BINOMIAL"
        elif modelo_atual == PricingModelType.MONTE_CARLO: metodologia_str = "MONTE_CARLO"
        elif modelo_atual == PricingModelType.RSU: metodologia_str = "COTACAO"

        # Lógica de Liquidação
        tipo_liq_analise = analysis.settlement_type
        forma_liq_str = "CAIXA" if tipo_liq_analise == SettlementType.CASH_SETTLED else ("ACOES" if emp_aberta else "CAIXA")

        # Consolida Inputs
        manual_inputs = {
            "empresa": {
                "nome": emp_nome, "ticker": emp_ticker, "capital_aberto": emp_aberta,
                "bolsa_nome": nome_bolsa
            },
            "programa": {
                "nome": prog_nome, "data_outorga": prog_data, 
                "qtd_beneficiarios": prog_qtd, 
                "metodologia": metodologia_str, "forma_liquidacao": forma_liq_str,
                "tipo_detalhado": tipo_detalhado
            },
            "responsavel": {"nome": resp_nome, "cargo": resp_cargo, "email": resp_email},
            "contab": {
                "taxa_turnover": turnover_contab, 
                "tem_encargos": tem_encargos,
                "tem_performance_nao_mercado": tem_nao_mercado,
                "percentual_atingimento": perc_atingimento 
            },
            "calculo_extra": {
                "metodo_privado": metodo_privado,
                "indice_correcao_nome": indice_texto,
                "moeda_selecionada": "BRL" if "BRL" in moeda_selecionada else "USD", 
                "cenario_dividendos": cenario_div.split(" ")[0] 
            }
        }

        # Busca template
        # Tenta caminhos relativos comuns
        possible_paths = ["templates/TEMPLATE_FINAL_PADRAO.docx", "TEMPLATE_FINAL_PADRAO.docx", "ui/TEMPLATE_FINAL_PADRAO.docx"]
        template_path = next((p for p in possible_paths if os.path.exists(p)), None)

        if not template_path:
            st.error("❌ Template 'TEMPLATE_FINAL_PADRAO.docx' não encontrado na pasta 'templates/'.")
            return

        with st.spinner("Compilando laudo..."):
            context = ReportService.generate_report_context(
                analysis,
                AppState.get_tranches(),
                results,
                manual_inputs
            )
            docx_bytes = ReportService.render_template(template_path, context)
        
        st.download_button(
            label=f"💾 Baixar Laudo: {emp_nome}.docx",
            data=docx_bytes,
            file_name=f"Laudo_{emp_nome.replace(' ', '_')}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
        st.success("Laudo gerado com sucesso!")
        
    except Exception as e:
        st.error(f"Erro ao gerar documento: {str(e)}")
