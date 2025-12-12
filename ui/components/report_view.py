import streamlit as st
import os
from datetime import date
from ui.state import AppState
from services.report_service import ReportService

def render_report_tab():
    results = AppState.get_calc_results()
    analysis = AppState.get_analysis()
    
    if not results:
        st.warning("⚠️ Realize o cálculo na aba anterior primeiro.")
        return

    st.header("Gerador de Laudo (CPC 10)")
    
    with st.form("report_form"):
        st.subheader("Dados Corporativos")
        col1, col2 = st.columns(2)
        emp_nome = col1.text_input("Empresa", "Minha Empresa S.A.")
        ticker = col2.text_input("Ticker", "TICK3")
        
        st.subheader("Dados do Programa")
        prog_nome = st.text_input("Nome do Plano", "Plano de ILP 2024")
        
        submitted = st.form_submit_button("Gerar Documento")
        
        if submitted:
            _generate_docx(emp_nome, ticker, prog_nome, analysis, results)

def _generate_docx(emp_nome, ticker, prog_nome, analysis, results):
    """Compila o contexto e chama o serviço de relatório."""
    
    # Monta o dicionário manual que o ReportService espera
    manual_inputs = {
        "empresa": {"nome": emp_nome, "ticker": ticker, "capital_aberto": True},
        "programa": {
            "nome": prog_nome, 
            "data_outorga": date.today(),
            "metodologia": analysis.model_recommended.value if analysis else "N/A"
        },
        "responsavel": {"nome": "Gerado por Icarus AI"},
        "contab": {"tem_encargos": False},
        "calculo_extra": {"cenario_dividendos": "ZERO"}
    }
    
    try:
        # Busca template padrão
        template_path = "templates/TEMPLATE_FINAL_PADRAO.docx"
        if not os.path.exists(template_path):
            st.error(f"Template não encontrado em: {template_path}")
            return

        context = ReportService.generate_report_context(
            analysis,
            AppState.get_tranches(),
            results,
            manual_inputs
        )
        
        docx = ReportService.render_template(template_path, context)
        
        st.download_button(
            label="💾 Baixar Laudo .docx",
            data=docx,
            file_name=f"Laudo_{emp_nome}.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        )
    except Exception as e:
        st.error(f"Erro na geração: {str(e)}")
