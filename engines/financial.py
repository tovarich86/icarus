"""
Módulo de Motores Financeiros (Engines) - IFRS 2 & Atuária
Versão Corrigida: Juros Contínuos, Proteção M, Estabilidade CRR Dinâmica.
"""
import numpy as np
import math
from numba import jit
from scipy.stats import norm

EPSILON = 1e-9

@jit(nopython=True, fastmath=True)
def _numba_norm_cdf(x):
    return 0.5 * (1 + math.erf(x / 1.41421356))

@jit(nopython=True, fastmath=True)
def _calculate_lockup_discount_numba(volatility, lockup_time, stock_price, q):
    if lockup_time <= EPSILON: return 0.0
    
    # Cast para float para evitar erro de tipo no Numba
    vol = float(volatility)
    t = float(lockup_time)
    s = float(stock_price)
    yld = float(q)
    
    vol_sq_t = (vol ** 2) * t
    term_inner = 2 * (np.exp(vol_sq_t) - vol_sq_t - 1)
    if term_inner <= EPSILON: return 0.0
    val_log = np.exp(vol_sq_t) - 1
    if val_log <= EPSILON: return 0.0
    b = vol_sq_t + np.log(term_inner) - 2 * np.log(val_log)
    if b < 0: return 0.0
    a = np.sqrt(b)
    
    return s * np.exp(-yld * t) * (_numba_norm_cdf(a/2) - _numba_norm_cdf(-a/2))

class FinancialMath:
    @staticmethod
    def calculate_lockup_discount(volatility, lockup_time, stock_price, q=0.0):
        return _calculate_lockup_discount_numba(float(volatility), float(lockup_time), float(stock_price), float(q))

    @staticmethod
    def bs_call(S, K, T, r_effective, sigma, q=0.0):
        # 1. Definição explícita de variáveis (Fix NameError q_in)
        S, K, T, r_eff, sigma, q_input = float(S), float(K), float(T), float(r_effective), float(sigma), float(q)
        
        # 2. Correção Matemática: Converter Taxas Efetivas para Contínuas
        # Para Black-Scholes, o input costuma ser taxa anual efetiva.
        r = np.log(1 + r_eff)
        q_rate = np.log(1 + q_input)

        if T <= EPSILON: return max(S - K, 0.0)
        if sigma <= EPSILON:
            val = (S * np.exp(-q_rate * T)) - (K * np.exp(-r * T))
            return max(val, 0.0)
        if K <= EPSILON: return S * np.exp(-q_rate * T)
        if S <= EPSILON: return 0.0

        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r - q_rate + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        return S * np.exp(-q_rate * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def binomial_custom_optimized(
        S, K, r_effective, vol, q_yield_eff, 
        vesting_years, turnover_w, multiple_M, hurdle_H, 
        T_years, inflacao_anual, lockup_years, tipo_exercicio=0
    ):
        # 1. Correção da Taxa para Contínua (Juros e Dividendos)
        r = np.log(1 + r_effective)
        q = np.log(1 + q_yield_eff) # Aqui usamos o argumento direto

        # 2. Proteção de Volatilidade Zero
        if vol < 1e-5:
            K_adj = K * ((1 + inflacao_anual)**T_years)
            return max((S * np.exp(-q * T_years)) - (K_adj * np.exp(-r * T_years)), 0.0)

        if T_years <= 1e-5: return max(S - K, 0.0)

        # --- CORREÇÃO 3: Estabilidade Dinâmica da Árvore (CRR) ---
        base_steps = int(T_years * 252)
        
        # Condition: dt < sigma^2 / (r-q)^2
        # Garante probabilidade 'p' entre 0 e 1 naturalmente
        min_steps_stability = 50
        denom_stability = (r - q)**2
        
        if denom_stability > 1e-9 and vol > 1e-9:
             calc_steps = int(T_years * denom_stability / (vol**2)) + 1
             if calc_steps > min_steps_stability:
                 min_steps_stability = calc_steps

        total_steps = base_steps
        if total_steps < min_steps_stability:
            total_steps = min_steps_stability
            
        # Soft caps para performance
        if total_steps > 5000: total_steps = 5000
        if total_steps < 50: total_steps = 50
        
        dt = T_years / total_steps
        Nv = int(vesting_years / dt)
        
        u = np.exp(vol * np.sqrt(dt))
        d = 1.0 / u
        
        denom = u - d
        if denom == 0: denom = 1e-9
        
        p = (np.exp((r - q) * dt) - d) / denom
        
        # Clamp de segurança final (apenas erros de arredondamento)
        if p < 0.0: p = 0.0
        if p > 1.0: p = 1.0
        
        prob_ficar = np.exp(-turnover_w * dt)
        prob_sair = 1.0 - prob_ficar

        # Inicialização (Vencimento)
        j_idx = np.arange(total_steps + 1)
        ST = S * (u ** (total_steps - j_idx)) * (d ** j_idx)
        K_final = K * ((1 + inflacao_anual)**T_years)
        
        vals_base = ST.copy()
        if lockup_years > 0:
            for i in range(len(vals_base)):
                disc = _calculate_lockup_discount_numba(vol, lockup_years, vals_base[i], q)
                vals_base[i] -= disc
        
        payoffs = np.maximum(vals_base - K_final, 0.0)
        option_values = np.where(ST >= hurdle_H, payoffs, 0.0)
        
        # Indução Retroativa
        for i in range(total_steps - 1, -1, -1):
            # Valor de "Esperar" (Hold)
            hold_values = np.exp(-r * dt) * (p * option_values[:-1] + (1 - p) * option_values[1:])
            
            # Se for antes do vesting, só existe risco de saída (forfeiture)
            if i < Nv:
                option_values = prob_ficar * hold_values
                continue
                
            # Pós-Vesting: Decisão de Exercício
            time_elapsed = i * dt
            j_grid = np.arange(i + 1)
            S_node = S * (u ** (i - j_grid)) * (d ** j_grid)
            K_curr = K * ((1 + inflacao_anual)**time_elapsed)

            # Valor Intrínseco (Exercer Agora)
            S_exerc = S_node.copy()
            if lockup_years > 0:
                for k in range(len(S_exerc)):
                    S_exerc[k] -= _calculate_lockup_discount_numba(vol, lockup_years, S_exerc[k], q)
            intrinsic_value = np.maximum(S_exerc - K_curr, 0.0)
            
            # --- Lógica do Múltiplo M ---
            if multiple_M >= 1.0:
                # Hull-White: Exerce se S >= M * K
                force_exercise_mask = (S_node >= (multiple_M * K_curr)) & (tipo_exercicio == 0)
            else:
                force_exercise_mask = np.zeros_like(S_node, dtype=np.bool_)

            value_node_stay = (prob_sair * intrinsic_value) + (prob_ficar * hold_values)
            
            if tipo_exercicio == 0: # Americano
                rational_exercise = np.maximum(value_node_stay, intrinsic_value)
                final_node_val = np.where(force_exercise_mask, intrinsic_value, rational_exercise)
            else:
                final_node_val = value_node_stay

            hurdle_mask = (S_node >= hurdle_H)
            option_values = np.where(hurdle_mask, final_node_val, prob_ficar * hold_values)
        
        return option_values[0]
