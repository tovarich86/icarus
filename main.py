"""
Ponto de entrada da aplicação Icarus.
Execute com: streamlit run main.py
"""

import streamlit as st
from ui.app_interface import IFRS2App

# Configuração da página deve ser a PRIMEIRA instrução Streamlit
st.set_page_config(
    page_title="Icarus: Beta Modular",
    layout="wide",
    page_icon="🛡️"
)

if __name__ == "__main__":
    app = IFRS2App()
    app.run()
