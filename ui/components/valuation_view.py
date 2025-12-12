import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta

from ui.state import AppState
from core.domain import PricingModelType, Tranche
from engines.financial import FinancialMath
from services.market_data import MarketDataService

def render_valuation_dashboard():
    analysis = AppState.get_analysis()
    
    if not analysis:
        st.info("👈 Faça o upload do contrato na barra lateral para iniciar.")
        return

    # --- Seção 1: Diagnóstico (Resumo do que a IA achou) ---
    with st.container(border=True):
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("Diagnóstico do Plano")
            st.info(f"**Resumo:** {analysis.summary}")
            st.caption(f"**Classificação Contábil:** {analysis.settlement_type.value}")
        with c2:
            st.subheader("Metodologia")
            st.success(f"**Modelo Recomendado:** {analysis.model_recommended.value}")
            st.markdown(f"> {analysis.methodology_rationale}")

    st.divider()

    # --- Seção 2: Inputs de Mercado Globais ---
    st.subheader("Premissas de Mercado")
    c1, c2, c3, c4 = st.columns(4)
    S = c1.number_input("Spot (R$)", value=50.0, step=0.5, format="%.2f")
    K = c2.number_input("Strike (R$)", value=analysis.strike_price, step=0.5, format="%.2f")
    vol_input = c3.number_input("Volatilidade (% a.a.)", value=30.0, step=1.0) / 100
    r_input = c4.number_input("Taxa Livre de Risco (% a.a.)", value=10.75, step=0.1) / 100
    
    q_input = st.number_input("Dividend Yield (% a.a.)", value=0.0, step=0.1) / 100

    # Seletor de Modelo (Permite Override do usuário)
    active_model = st.selectbox(
        "Modelo de Precificação (Override)", 
        options=[m.value for m in PricingModelType],
        index=list(PricingModelType).index(analysis.model_recommended) if analysis.model_recommended in list(PricingModelType) else 0
    )

    st.divider()

    # --- Seção 3: Tranches e Cálculo ---
    st.subheader("Estrutura de Vesting")
    _render_tranches_editor()

    if st.button("🧮 Calcular Fair Value", type="primary", use_container_width=True):
        _execute_calculation(S, K, vol_input, r_input, q_input, active_model)

def _render_tranches_editor():
    """Tabela editável de tranches."""
    tranches = AppState.get_tranches()
    
    # Converte Pydantic models para lista de dicts para o DataEditor
    data = [t.model_dump() for t in tranches]
    
    edited_data = st.data_editor(
        data, 
        column_config={
            "vesting_date": st.column_config.NumberColumn("Vesting (Anos)", format="%.2f"),
            "proportion": st.column_config.NumberColumn("Peso (0-1)", format="%.2f"),
            "expiration_date": st.column_config.NumberColumn("Vencimento (Anos)", format="%.2f"),
            "custom_strike": st.column_config.NumberColumn("Strike Custom", format="%.2f"),
        },
        num_rows="dynamic",
        use_container_width=True
    )

    # Atualiza o estado se houver mudança
    if edited_data != data:
        new_tranches = [Tranche(**row) for row in edited_data]
        AppState.set_tranches(new_tranches)

def _execute_calculation(S, K, vol, r, q, model_name):
    """Orquestra o cálculo financeiro e salva no estado."""
    tranches = AppState.get_tranches()
    results = []
    total_fv = 0.0

    for i, t in enumerate(tranches):
        # Definição de Parâmetros por Tranche
        T_life = t.expiration_date if t.expiration_date else 10.0
        K_final = t.custom_strike if t.custom_strike is not None else K
        
        fv_unit = 0.0
        
        # Dispatch simples de modelos (Pode ser expandido depois)
        if "Black-Scholes" in model_name:
            fv_unit = FinancialMath.bs_call(S, K_final, T_life, r, vol, q)
        elif "RSU" in model_name:
            # Simplificação RSU: Spot descontado
            fv_unit = S * np.exp(-q * t.vesting_date) 
        else:
            # Fallback para BS
            fv_unit = FinancialMath.bs_call(S, K_final, T_life, r, vol, q)

        fv_weighted = fv_unit * t.proportion
        total_fv += fv_weighted
        
        results.append({
            "Tranche": i + 1,
            "Vesting": t.vesting_date,
            "FV Unit": fv_unit,
            "FV Ponderado": fv_weighted,
            "S": S, "K": K_final, "Vol": vol, "r": r, "T": T_life, "q": q
        })

    # Persistência
    AppState.set_calc_results(results)
    
    # Exibição
    st.success(f"Cálculo Realizado! Fair Value Total: R$ {total_fv:,.2f}")
    st.dataframe(pd.DataFrame(results))
