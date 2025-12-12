import streamlit as st
import pandas as pd
import numpy as np
import io
import sys
from datetime import date, timedelta

from ui.state import AppState
from core.domain import PricingModelType, Tranche
from engines.financial import FinancialMath
from services.market_data import MarketDataService
from services.ai_service import DocumentService

def render_valuation_dashboard():
    analysis = AppState.get_analysis()
    
    if not analysis:
        st.info("👈 Faça o upload do contrato na barra lateral para iniciar.")
        return

    # --- Seção 1: Diagnóstico ---
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

    # --- Seção 2: Inputs Globais e Mercado ---
    st.subheader("Premissas de Mercado")
    c1, c2, c3, c4 = st.columns(4)
    S = c1.number_input("Spot (R$)", value=50.0, step=0.5, format="%.2f", help="Preço atual da ação.")
    K = c2.number_input("Strike (R$)", value=analysis.strike_price, step=0.5, format="%.2f", help="Preço de exercício.")
    
    # Widget Avançado de Volatilidade
    with c3:
        vol_input = _render_volatility_widget_global()
    
    # Widget Avançado de Taxa de Juros
    with c4:
        r_input = _render_rate_widget_global()
    
    q_input = st.number_input("Dividend Yield (% a.a.)", value=0.0, step=0.1) / 100

    # Override de Modelo
    opts = [m for m in PricingModelType if m != PricingModelType.UNDEFINED]
    idx = opts.index(analysis.model_recommended) if analysis.model_recommended in opts else 0
    active_model = st.selectbox(
        "Modelo de Precificação (Override)", 
        options=opts,
        index=idx,
        format_func=lambda x: x.value
    )

    st.divider()

    # --- Seção 3: Renderização Específica por Modelo ---
    if active_model == PricingModelType.MONTE_CARLO:
        _render_monte_carlo_ai_section(S, K, r_input, vol_input, q_input, analysis)
    else:
        # Para modelos determinísticos (BS, Binomial, RSU)
        st.subheader("Estrutura de Vesting")
        
        # Inputs Específicos do Binomial
        binomial_params = {}
        if active_model == PricingModelType.BINOMIAL:
            with st.expander("⚙️ Parâmetros Binomiais (Turnover & Barreiras)", expanded=True):
                bc1, bc2, bc3 = st.columns(3)
                binomial_params['turnover'] = bc1.number_input("Turnover (% a.a.)", value=analysis.turnover_rate*100) / 100
                binomial_params['multiple_m'] = bc2.number_input("Múltiplo M (Early Exercise)", value=analysis.early_exercise_multiple)
                binomial_params['strike_corr'] = bc3.number_input("Correção Strike (% a.a.)", value=4.5 if analysis.has_strike_correction else 0.0) / 100
        
        # Editor de Tranches
        _render_tranches_editor(active_model)

        if st.button("🧮 Calcular Fair Value", type="primary", use_container_width=True):
            _execute_deterministic_calc(S, K, vol_input, r_input, q_input, active_model, binomial_params)

# --- Sub-componentes de Mercado (Restaurados do app_interface original) ---

def _render_volatility_widget_global():
    """Widget com Popover para busca de Volatilidade (Yahoo Finance)."""
    key_val = "global_vol_val"
    if key_val not in st.session_state: st.session_state[key_val] = 30.0

    c_in, c_pop = st.columns([0.85, 0.15])
    val = c_in.number_input("Volatilidade (%)", value=st.session_state[key_val], step=1.0) / 100
    
    with c_pop.popover("🔍"):
        st.markdown("###### Buscar Volatilidade")
        tk = st.text_input("Tickers (ex: VALE3)", "VALE3")
        d1 = st.date_input("Início", date.today()-timedelta(days=365))
        if st.button("Buscar Dados"):
            with st.spinner("Consultando..."):
                res = MarketDataService.get_peer_group_volatility([t.strip() for t in tk.split(',')], d1, date.today())
                if "summary" in res:
                    new_vol = res['summary']['mean_ewma'] * 100
                    st.session_state[key_val] = new_vol
                    st.success(f"Vol EWMA encontrada: {new_vol:.2f}%")
                    st.rerun()
    return val

def _render_rate_widget_global():
    """Widget com Popover para busca de Taxa DI (B3)."""
    key_val = "global_rate_val"
    if key_val not in st.session_state: st.session_state[key_val] = 10.75

    c_in, c_pop = st.columns([0.85, 0.15])
    val = c_in.number_input("Taxa Livre Risco (%)", value=st.session_state[key_val], step=0.1) / 100
    
    with c_pop.popover("📉"):
        st.markdown("###### Curva DI (B3)")
        if st.button("Carregar B3"):
            with st.spinner("Lendo B3..."):
                df = MarketDataService.get_di_data_b3(date.today())
                if not df.empty:
                    st.session_state['di_data_cache'] = df
        
        if 'di_data_cache' in st.session_state:
            df = st.session_state['di_data_cache']
            st.dataframe(df[['Vencimento_Fmt', 'Taxa']], hide_index=True, use_container_width=True)
            
            # Seletor simples
            opts = [f"{row['Vencimento_Fmt']} - {row['Taxa']*100:.2f}%" for _, row in df.iterrows()]
            sel = st.selectbox("Selecionar Vértice", opts)
            if st.button("Aplicar Taxa"):
                rate_val = float(sel.split(' - ')[1].replace('%',''))
                st.session_state[key_val] = rate_val
                st.rerun()
    return val

# --- Sub-componentes de Lógica de Negócio ---

def _render_tranches_editor(model_type):
    """Editor de tranches adaptado para o Streamlit."""
    tranches = AppState.get_tranches()
    data = [t.model_dump() for t in tranches]
    
    cols = {
        "vesting_date": st.column_config.NumberColumn("Vesting (Anos)", format="%.2f"),
        "proportion": st.column_config.NumberColumn("Peso (0-1)", format="%.2f"),
        "expiration_date": st.column_config.NumberColumn("Vencimento (Anos)", format="%.2f"),
        "custom_rate": st.column_config.NumberColumn("Taxa (%)", format="%.4f"),
        "custom_strike": st.column_config.NumberColumn("Strike Custom", format="%.2f")
    }
    
    # Mostra coluna de Strike Custom apenas se necessário (Binomial/BS)
    if model_type != PricingModelType.RSU:
        cols["custom_strike"] = st.column_config.NumberColumn("Strike Custom", format="%.2f")

    edited_data = st.data_editor(
        data, 
        column_config=cols,
        num_rows="dynamic",
        use_container_width=True,
        key=f"editor_{model_type}"
    )

    if edited_data != data:
        # Reconstrói objetos Pydantic a partir do dict
        try:
            new_tranches = [Tranche(**row) for row in edited_data]
            AppState.set_tranches(new_tranches)
        except Exception:
            pass # Evita erro durante a digitação incompleta

def _render_monte_carlo_ai_section(S, K, r, vol, q, analysis):
    """Restaura a funcionalidade de geração e execução de código Python."""
    st.info("🤖 Monte Carlo via IA: Gera e executa script customizado.")
    
    tranches_dates = [t.vesting_date for t in AppState.get_tranches()]
    params = {
        "S0": S, "K": K, "r": r, "sigma": vol, "q": q,
        "T": analysis.option_life_years,
        "vesting_schedule": tranches_dates
    }
    
    c1, c2 = st.columns(2)
    # Passo 1: Gerar
    if c1.button("1. Gerar Código Python"):
        api_key = st.secrets.get("GEMINI_API_KEY", "") # Idealmente buscar do state/sidebar
        if not api_key:
            st.error("API Key necessária para gerar código.")
            return
            
        with st.spinner("Escrevendo script..."):
            ctx = AppState.get_context_text()
            code = DocumentService.generate_custom_monte_carlo_code(ctx, params, api_key)
            AppState.set_mc_code(code)
            
    # Passo 2: Editar e Executar
    current_code = AppState.get_mc_code()
    if current_code:
        edited_code = st.text_area("Script Python (Editável)", value=current_code, height=300)
        AppState.set_mc_code(edited_code)
        
        if c2.button("2. Executar Simulação", type="primary"):
            _run_custom_code(edited_code)

def _run_custom_code(code):
    """Execução segura de código (Sandbox simulado)."""
    old_stdout = io.StringIO()
    sys.stdout = old_stdout
    local_scope = {}
    
    try:
        with st.spinner("Simulando..."):
            exec(code, local_scope)
            
        output = old_stdout.getvalue()
        sys.stdout = sys.__stdout__
        
        st.text("Output do Console:")
        st.code(output)
        
        if 'fv' in local_scope:
            fv = float(local_scope['fv'])
            st.metric("Fair Value (Resultado)", f"R$ {fv:,.2f}")
            # Salva um resultado sintético para o relatório
            AppState.set_calc_results([{
                "Tranche": "Total (MC)", "FV Unit": fv, "FV Ponderado": fv,
                "S": 0, "K": 0, "Vol": 0, "r": 0, "T": 0, "q": 0
            }])
        else:
            st.warning("Variável 'fv' não encontrada no escopo final.")
            
    except Exception as e:
        sys.stdout = sys.__stdout__
        st.error(f"Erro na execução: {e}")

def _execute_deterministic_calc(S, K, vol, r, q, model, bin_params):
    """Calculadora para modelos fechados (BS, Binomial, RSU)."""
    tranches = AppState.get_tranches()
    results = []
    total_fv = 0.0

    for i, t in enumerate(tranches):
        T_life = t.expiration_date if t.expiration_date else 5.0
        K_final = t.custom_strike if t.custom_strike is not None else K
        r_calc = t.custom_rate if t.custom_rate is not None else r_global
        
        fv = 0.0
        
        if model == PricingModelType.BLACK_SCHOLES_GRADED:
            fv = FinancialMath.bs_call(S, K_final, T_life, r_calc, vol, q)
            
        elif model == PricingModelType.RSU:
            base_val = S * np.exp(-q * t.vesting_date)
            lockup = AppState.get_analysis().lockup_years
            discount = 0.0
            if lockup > 0:
                discount = FinancialMath.calculate_lockup_discount(vol, lockup, base_val, q) # Chaffe usa r? (Verificar engine)
            fv = base_val - discount
            
        elif model == PricingModelType.BINOMIAL:
            # Binomial Completo
            fv = FinancialMath.binomial_custom_optimized(
                S=S, K=K_final, r_effective=r_calc, vol=vol, q_yield_eff=q,
                vesting_years=t.vesting_date,
                turnover_w=bin_params.get('turnover', 0.0),
                multiple_M=bin_params.get('multiple_m', 2.0),
                hurdle_H=0.0, # Hurdle não exposto na UI simplificada
                T_years=T_life,
                inflacao_anual=bin_params.get('strike_corr', 0.0),
                lockup_years=AppState.get_analysis().lockup_years
            )

        w_fv = fv * t.proportion
        total_fv += w_fv
        
        results.append({
            "Tranche": i + 1,
            "Vesting": t.vesting_date,
            "Vencimento": T_life,
            "FV Unit": fv,
            "FV Ponderado": w_fv,
            "S": S, "K": K_final, "Vol": vol, "r": r, "q": q
        })

    AppState.set_calc_results(results)
    st.success(f"Cálculo Concluído! Fair Value Total: R$ {total_fv:,.2f}")
    st.dataframe(pd.DataFrame(results))
