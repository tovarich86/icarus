import streamlit as st
from ui.state import AppState
from ui.components.sidebar import render_sidebar
from ui.components.valuation_view import render_valuation_dashboard
from ui.components.report_view import render_report_tab

class IFRS2App:
    """
    Aplicação Principal (Orquestrador).
    Responsável apenas pelo layout macro e injeção de dependências.
    """
    
    def run(self):
        # 1. Configuração Global da Página
        # Deve ser a primeira chamada Streamlit no script principal (ou aqui se chamado diretamente)
        # Nota: Se o app.py já chamar set_page_config, esta linha pode ser redundante, 
        # mas mantê-la aqui garante que funcione se rodar este arquivo diretamente.
        try:
            st.set_page_config(page_title="Icarus Valuation", layout="wide", page_icon="🛡️")
        except:
            pass # Ignora se já tiver sido configurado no app.py

        # 2. Inicializa o ViewModel (Session State)
        AppState.initialize()

        # 3. Renderiza a Barra Lateral (Inputs e IA)
        render_sidebar()

        # 4. Layout Principal
        st.title("🛡️ Icarus: Valuation IFRS 2 (AI-Native)")
        
        # Abas Superiores
        tab_calc, tab_laudo = st.tabs(["🧮 1. Cálculo & Valuation", "📝 2. Gerador de Laudo"])

        with tab_calc:
            render_valuation_dashboard()

        with tab_laudo:
            render_report_tab()

if __name__ == "__main__":
    app = IFRS2App()
    app.run()
