import streamlit as st
from services.ai_service import DocumentService
from services.strategy import ModelSelectorService
from ui.state import AppState

def render_sidebar():
    with st.sidebar:
        st.header("1. Documentação")
        
        # Gestão de Segredos vs Input Manual
        if "GEMINI_API_KEY" in st.secrets:
            gemini_key = st.secrets["GEMINI_API_KEY"]
            st.success("🔑 API Key detectada")
        else:
            gemini_key = st.text_input("Gemini API Key", type="password")
        
        st.subheader("Upload de Contratos")
        uploaded_files = st.file_uploader(
            "PDF ou DOCX", 
            type=['pdf', 'docx'], 
            accept_multiple_files=True
        )
        
        manual_text = st.text_area(
            "Descrição Manual", 
            height=100, 
            placeholder="Cole cláusulas aqui ou digite regras (ex: Vesting 3 anos)..."
        )
        
        use_ai = st.toggle(
            "Usar IA Generativa", 
            value=True, 
            help="Se desligado, usa apenas regras (Regex). Mais rápido, mas menos detalhado."
        )
        
        if st.button("🚀 Analisar Contrato", type="primary", use_container_width=True):
            _handle_analysis(uploaded_files, manual_text, gemini_key, use_ai)
            
        st.divider()
        
        if st.button("🛠️ Modo Manual (Reset)", type="secondary", use_container_width=True):
            AppState.enable_manual_mode()
            st.rerun()
            
        st.caption("v.2.0 AI-Native Core")

def _handle_analysis(uploaded_files, manual_text, api_key, use_ai):
    """Lógica interna de processamento."""
    combined_text = ""
    
    if uploaded_files:
        with st.spinner("Lendo arquivos..."):
            for f in uploaded_files:
                combined_text += f"--- {f.name} ---\n{DocumentService.extract_text(f)}\n"
    
    if manual_text: 
        combined_text += f"--- MANUAL ---\n{manual_text}"
    
    if not combined_text.strip():
        st.error("Forneça um arquivo ou texto.")
        return

    # Salva o texto bruto no estado para uso posterior (ex: Monte Carlo)
    AppState.set_context_text(combined_text)

    with st.spinner("🔍 Executando Análise Híbrida (Regras + IA)..."):
        try:
            # 1. Análise (Extração)
            analysis = DocumentService.analyze_plan_hybrid(
                text=combined_text, 
                api_key=api_key, 
                use_ai=use_ai
            )
            
            # 2. Estratégia (Seleção de Modelo)
            analysis = ModelSelectorService.select_model(analysis)
            
            # 3. Persistência no Estado (ViewModel)
            AppState.set_analysis(analysis)
            
            st.success("Análise concluída!")
            st.rerun()
            
        except Exception as e:
            st.error(f"Erro crítico na análise: {str(e)}")
