import streamlit as st
import pandas as pd
import numpy as np
import io
import sys
from datetime import date, timedelta

from ui.state import AppState
from core.domain import PricingModelType, Tranche, SettlementType
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
            st.caption(f"**Classificação Contábil:** {str(analysis.settlement_type)}")
        with c2:
            st.subheader("Metodologia")
            st.success(f"**Modelo Recomendado:** {str(analysis.model_recommended)}")
            st.markdown(f"> {analysis.methodology_rationale}")

    st.divider()

    # --- Seção 2: Premissas de Mercado (Globais/Defaults) ---
    st.subheader("Premissas de Mercado (Referência Global)")
    c1, c2, c3, c4 = st.columns(4)
    
    S_global = c1.number_input("Spot (R$)", value=50.0, step=0.5, format="%.2f", help="Preço atual da ação.", key="glob_S")
    K_global = c2.number_input("Strike (R$)", value=analysis.strike_price, step=0.5, format="%.2f", help="Preço de exercício global.", key="glob_K")
    
    with c3:
        vol_global = _render_volatility_widget_global()
    with c4:
        r_global = _render_rate_widget_global()
    
    q_global = st.number_input("Dividend Yield (% a.a.)", value=0.0, step=0.1, key="glob_q") / 100

    # Override de Modelo
    opts = [m for m in PricingModelType if m != PricingModelType.UNDEFINED]
    idx = opts.index(analysis.model_recommended) if analysis.model_recommended in opts else 0
    active_model = st.selectbox(
        "Modelo de Precificação (Override)", 
        options=opts,
        index=idx,
        format_func=lambda x: x.value
    )
    
    # Sincroniza modelo escolhido com o objeto de análise para o relatório
    if active_model != analysis.model_recommended:
        analysis.model_recommended = active_model

    st.divider()

    # --- Seção 3: Renderização Específica por Modelo ---
    if active_model == PricingModelType.MONTE_CARLO:
        _render_monte_carlo_ai_section(S_global, K_global, r_global, vol_global, q_global, analysis)
    else:
        # VISUALIZAÇÃO DETALHADA RESTAURADA (Botões + Cards)
        _render_detailed_tranches_view(active_model, S_global, K_global, vol_global, r_global, q_global, analysis)


def _render_detailed_tranches_view(model, S, K, vol, r, q, analysis):
    st.subheader("Estrutura de Vesting & Precificação")
    
    # 1. BOTÕES DE CONTROLE RESTAURADOS
    c_add, c_rem, _ = st.columns([1, 1, 3])
    if c_add.button("➕ Adicionar Tranche", use_container_width=True):
        AppState.add_tranche_action()
        st.rerun()
    if c_rem.button("➖ Remover Tranche", use_container_width=True):
        AppState.remove_last_tranche_action()
        st.rerun()

    tranches = AppState.get_tranches()
    inputs_calc = []
    
    if not tranches:
        st.warning("Nenhuma tranche definida. Adicione uma para começar.")
        return

    # 2. RENDERIZAÇÃO CARTÃO POR CARTÃO
    for i, t in enumerate(tranches):
        with st.container(border=True):
            st.markdown(f"##### 🔹 Tranche {i+1}")
            
            # --- Linha 1: Tempos e Proporção ---
            c1, c2, c3, c4 = st.columns(4)
            
            def_exp = t.expiration_date if t.expiration_date else analysis.option_life_years
            t_exp = c1.number_input("Vencimento (Anos)", value=float(def_exp), key=f"t_exp_{i}", min_value=0.1)
            
            t_vest = c2.number_input("Vesting (Anos)", value=float(t.vesting_date), key=f"t_vest_{i}", min_value=0.0)
            
            t_prop = c3.number_input("Peso (%)", value=float(t.proportion*100), key=f"t_prop_{i}", step=5.0) / 100
            
            # --- Linha 2: Parâmetros Específicos (Vol/Rate/Strike) ---
            cm1, cm2, cm3 = st.columns(3)
            
            # Strike Específico (Override)
            k_display = t.custom_strike if t.custom_strike is not None else K
            t_k = cm1.number_input("Strike", value=float(k_display), key=f"t_k_{i}")
            
            # Widgets Locais de Vol e Rate (Restaurados)
            with cm2:
                # Usa Session State local para overrides sem persistir no objeto Tranche (que é limpo)
                key_vol = f"vol_local_{i}"
                if key_vol not in st.session_state: st.session_state[key_vol] = vol * 100
                t_vol = _render_local_vol_widget(i, st.session_state[key_vol]) / 100
            
            with cm3:
                key_rate = f"rate_local_{i}"
                if key_rate not in st.session_state: st.session_state[key_rate] = r * 100
                t_r = _render_local_rate_widget(i, st.session_state[key_rate], t_exp) / 100

            # --- Linha 3: Avançado (Binomial/Lockup) ---
            # Defaults da Análise
            t_lock = analysis.lockup_years
            t_turnover = analysis.turnover_rate
            t_m = analysis.early_exercise_multiple
            t_corr = 4.5 if analysis.has_strike_correction else 0.0

            if model == PricingModelType.BINOMIAL or model == PricingModelType.RSU:
                 with st.expander("⚙️ Avançado (Lockup, Turnover, Barreiras)", expanded=False):
                     ca1, ca2, ca3, ca4 = st.columns(4)
                     t_lock = ca1.number_input("Lockup (Anos)", value=float(t_lock), key=f"t_lock_{i}")
                     
                     if model == PricingModelType.BINOMIAL:
                         t_turnover = ca2.number_input("Turnover (% a.a.)", value=float(t_turnover*100), key=f"t_turn_{i}") / 100
                         t_m = ca3.number_input("Múltiplo M (Exer.)", value=float(t_m), key=f"t_m_{i}")
                         t_corr = ca4.number_input("Corr. Strike (% a.a.)", value=float(t_corr), key=f"t_corr_{i}") / 100
            
            # Consolida input para cálculo
            inputs_calc.append({
                "TrancheID": i+1,
                "S": S, "K": t_k, "q": q,
                "T": t_exp, "Vesting": t_vest, "Prop": t_prop,
                "Vol": t_vol, "r": t_r,
                "Lockup": t_lock, "Turnover": t_turnover, "M": t_m, "StrikeCorr": t_corr
            })

    # 3. BOTÃO DE CÁLCULO
    if st.button("🧮 Calcular Fair Value (Todos)", type="primary", use_container_width=True):
        _execute_calc_restore(inputs_calc, model)


# --- WIDGETS AUXILIARES (LOCAIS POR TRANCHE) ---

def _render_local_vol_widget(i, default_val):
    c_in, c_pop = st.columns([0.85, 0.15])
    key_w = f"w_vol_local_{i}"
    val = c_in.number_input("Vol (%)", value=float(default_val), key=key_w, step=1.0)
    
    with c_pop.popover("🔍"):
        st.markdown("###### Volatilidade Histórica")
        tk = st.text_input("Ticker", "VALE3", key=f"tk_{i}")
        if st.button("Buscar", key=f"btn_vol_{i}"):
            res = MarketDataService.get_peer_group_volatility([tk], date.today()-timedelta(days=365), date.today())
            if "summary" in res:
                st.session_state[f"vol_local_{i}"] = res['summary']['mean_ewma'] * 100
                st.rerun()
    return val

def _render_local_rate_widget(i, default_val, t_years):
    c_in, c_pop = st.columns([0.85, 0.15])
    key_w = f"w_rate_local_{i}"
    val = c_in.number_input("Taxa (%)", value=float(default_val), key=key_w, step=0.1)
    
    with c_pop.popover("📉"):
        st.markdown("###### Curva DI")
        if st.button("Carregar B3", key=f"btn_di_{i}"):
            df = MarketDataService.get_di_data_b3(date.today())
            st.session_state['di_cache'] = df
        
        if 'di_cache' in st.session_state:
            df = st.session_state['di_cache']
            target_days = t_years * 365
            idx = (df['Dias_Corridos'] - target_days).abs().idxmin()
            opts = [f"{row['Vencimento_Fmt']} - {row['Taxa']*100:.2f}%" for _, row in df.iterrows()]
            
            sel = st.selectbox("Vértice", opts, index=int(idx), key=f"sel_di_{i}")
            if st.button("Aplicar", key=f"app_di_{i}"):
                st.session_state[f"rate_local_{i}"] = float(sel.split('-')[1].replace('%',''))
                st.rerun()
    return val

# --- LÓGICA DE CÁLCULO ---

def _execute_calc_restore(inputs, model):
    results = []
    total_fv = 0.0

    for item in inputs:
        # Extração
        S, K, T, r, vol, q = item['S'], item['K'], item['T'], item['r'], item['Vol'], item['q']
        vesting, prop = item['Vesting'], item['Prop']
        lockup = item['Lockup']
        
        fv = 0.0
        
        if model == PricingModelType.BLACK_SCHOLES_GRADED:
            fv = FinancialMath.bs_call(S, K, T, r, vol, q)
            
        elif model == PricingModelType.RSU:
            base_val = S * np.exp(-q * vesting) # Desconta dividendos até vesting
            disc = 0.0
            if lockup > 0:
                disc = FinancialMath.calculate_lockup_discount(vol, lockup, base_val, q)
            fv = base_val - disc
            
        elif model == PricingModelType.BINOMIAL:
            fv = FinancialMath.binomial_custom_optimized(
                S=S, K=K, r_effective=r, vol=vol, q_yield_eff=q,
                vesting_years=vesting,
                turnover_w=item['Turnover'],
                multiple_M=item['M'],
                hurdle_H=0.0,
                T_years=T,
                inflacao_anual=item['StrikeCorr'],
                lockup_years=lockup
            )

        w_fv = fv * prop
        total_fv += w_fv
        
        # Adiciona resultado enriquecido
        res_row = item.copy()
        res_row.update({
            "FV Unit": fv,
            "FV Ponderado": w_fv
        })
        results.append(res_row)

    AppState.set_calc_results(results)
    st.success(f"Cálculo Concluído! Fair Value Total: R$ {total_fv:,.2f}")
    
    # Exibe Tabela Resumo
    df = pd.DataFrame(results)
    cols_show = ["TrancheID", "FV Unit", "FV Ponderado", "S", "K", "Vol", "T"]
    st.dataframe(df[[c for c in cols_show if c in df.columns]], use_container_width=True)

# --- MANTER WIDGETS GLOBAIS ORIGINAIS PARA RETROCOMPATIBILIDADE ---
def _render_volatility_widget_global():
    key_val = "global_vol_val"
    if key_val not in st.session_state: st.session_state[key_val] = 30.0
    c_in, c_pop = st.columns([0.85, 0.15])
    val = c_in.number_input("Volatilidade (%)", value=st.session_state[key_val], step=1.0) / 100
    with c_pop.popover("🔍"):
        tk = st.text_input("Ticker Global", "VALE3")
        if st.button("Buscar Global"):
            res = MarketDataService.get_peer_group_volatility([tk], date.today()-timedelta(days=365), date.today())
            if "summary" in res:
                st.session_state[key_val] = res['summary']['mean_ewma'] * 100
                st.rerun()
    return val

def _render_rate_widget_global():
    key_val = "global_rate_val"
    if key_val not in st.session_state: st.session_state[key_val] = 10.75
    c_in, c_pop = st.columns([0.85, 0.15])
    val = c_in.number_input("Taxa (%)", value=st.session_state[key_val], step=0.1) / 100
    return val

def _render_monte_carlo_ai_section(S, K, r, vol, q, analysis):
    st.info("🤖 Monte Carlo via IA: Gera e executa script customizado.")
    tranches_dates = [t.vesting_date for t in AppState.get_tranches()]
    params = {
        "S0": S, "K": K, "r": r, "sigma": vol, "q": q,
        "T": analysis.option_life_years,
        "vesting_schedule": tranches_dates
    }
    c1, c2 = st.columns(2)
    if c1.button("1. Gerar Código Python"):
        api_key = st.secrets.get("GEMINI_API_KEY", "")
        if not api_key:
            st.error("API Key necessária.")
            return
        with st.spinner("Escrevendo script..."):
            ctx = AppState.get_context_text()
            code = DocumentService.generate_custom_monte_carlo_code(ctx, params, api_key)
            AppState.set_mc_code(code)
    current_code = AppState.get_mc_code()
    if current_code:
        edited_code = st.text_area("Script Python", value=current_code, height=300)
        AppState.set_mc_code(edited_code)
        if c2.button("2. Executar Simulação", type="primary"):
            _run_custom_code(edited_code)

def _run_custom_code(code):
    old_stdout = io.StringIO()
    sys.stdout = old_stdout
    local_scope = {}
    try:
        with st.spinner("Simulando..."):
            exec(code, local_scope)
        output = old_stdout.getvalue()
        sys.stdout = sys.__stdout__
        st.text("Output:")
        st.code(output)
        if 'fv' in local_scope:
            fv = float(local_scope['fv'])
            st.metric("Fair Value", f"R$ {fv:,.2f}")
            AppState.set_calc_results([{"Tranche": "Total (MC)", "FV Unit": fv, "FV Ponderado": fv, "S": 0, "K": 0, "Vol": 0, "r": 0, "T": 0, "q": 0}])
    except Exception as e:
        sys.stdout = sys.__stdout__
        st.error(f"Erro: {e}")
