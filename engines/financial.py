"""
Módulo de Motores Financeiros (Engines).
Otimizado com Numba para alta performance em simulações iterativas (Binomial/Lattice).
"""
import numpy as np
import math
from numba import jit
from scipy.stats import norm # Importação movida para o topo

# Constante para evitar divisão por zero
EPSILON = 1e-9

@jit(nopython=True, fastmath=True)
def _numba_norm_cdf(x):
    """Implementação da CDF Normal compatível com Numba (usando Error Function)."""
    return 0.5 * (1 + math.erf(x / 1.41421356))

@jit(nopython=True, fastmath=True)
def _calculate_lockup_discount_numba(volatility, lockup_time, stock_price, q):
    if lockup_time <= EPSILON: return 0.0
    
    vol_sq_t = (volatility ** 2) * lockup_time
    term_inner = 2 * (np.exp(vol_sq_t) - vol_sq_t - 1)
    
    if term_inner <= EPSILON: return 0.0
    
    val_log = np.exp(vol_sq_t) - 1
    if val_log <= EPSILON: return 0.0

    b = vol_sq_t + np.log(term_inner) - 2 * np.log(val_log)
    if b < 0: return 0.0
    
    a = np.sqrt(b)
    discount_val = stock_price * np.exp(-q * lockup_time) * (_numba_norm_cdf(a / 2) - _numba_norm_cdf(-a / 2))
    return discount_val

class FinancialMath:
    @staticmethod
    def calculate_lockup_discount(volatility, lockup_time, stock_price, q=0.0):
        return _calculate_lockup_discount_numba(float(volatility), float(lockup_time), float(stock_price), float(q))

    @staticmethod
    def bs_call(S, K, T, r, sigma, q=0.0):
        """
        Black-Scholes SEM try/except silencioso.
        """
        # Garante floats para evitar erros de numpy com strings
        S, K, T, r, sigma, q = float(S), float(K), float(T), float(r), float(sigma), float(q)

        # 1. Tratamento de Tempo
        if T <= EPSILON: return max(S - K, 0.0)
        
        # 2. Tratamento de Volatilidade Zero
        if sigma <= EPSILON:
            val = (S * np.exp(-q * T)) - (K * np.exp(-r * T))
            return max(val, 0.0)
            
        # 3. Tratamento de Strike Zero (RSU Simples)
        if K <= EPSILON:
            return S * np.exp(-q * T)
            
        if S <= EPSILON: return 0.0

        # Sem try/except: Se falhar, queremos ver o erro no Streamlit!
        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def binomial_custom_optimized(
        S, K, r, vol, q, 
        vesting_years, turnover_w, multiple_M, hurdle_H, 
        T_years, inflacao_anual, lockup_years, tipo_exercicio=0
    ):
        # --- PROTEÇÃO CONTRA DIVISÃO POR ZERO (FIX NAN) ---
        if vol < 1e-5:
            # Volatilidade zero quebra o modelo CRR. Retorna intrínseco.
            K_adj = K * ((1 + inflacao_anual)**T_years)
            val = (S * np.exp(-q * T_years)) - (K_adj * np.exp(-r * T_years))
            return max(val, 0.0)

        if T_years <= 1e-5:
             return max(S - K, 0.0)

        # --- Lógica Padrão ---
        total_steps = int(T_years * 252)
        if total_steps > 2000: total_steps = 2000
        if total_steps < 50: total_steps = 50
        
        dt = T_years / total_steps
        Nv = int(vesting_years / dt)
        
        u = np.exp(vol * np.sqrt(dt))
        d = 1.0 / u
        p = (np.exp((r - q) * dt) - d) / (u - d) # Aqui ocorria o NaN se u~d
        
        prob_ficar = np.exp(-turnover_w * dt)
        prob_sair = 1.0 - prob_ficar

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
        
        for i in range(total_steps - 1, -1, -1):
            time_elapsed = i * dt
            K_curr = K * ((1 + inflacao_anual)**time_elapsed)
            
            hold_values = np.exp(-r * dt) * (p * option_values[:-1] + (1 - p) * option_values[1:])
            
            # Otimização: Só calcula nós necessários se estivermos dentro do vesting ou perto do dinheiro
            if i >= Nv:
                j_grid = np.arange(i + 1)
                S_node = S * (u ** (i - j_grid)) * (d ** j_grid)

                S_exerc = S_node.copy()
                if lockup_years > 0:
                    for k in range(len(S_exerc)):
                        S_exerc[k] -= _calculate_lockup_discount_numba(vol, lockup_years, S_exerc[k], q)
                
                intrinsic_value = np.maximum(S_exerc - K_curr, 0.0)
                
                force_exercise_mask = (S_node >= (multiple_M * K_curr)) & (tipo_exercicio == 0)
                
                value_node = (prob_sair * intrinsic_value) + (prob_ficar * hold_values)
                
                if tipo_exercicio == 0:
                    rational_exercise = np.maximum(value_node, intrinsic_value)
                    final_node_val = np.where(force_exercise_mask, intrinsic_value, rational_exercise)
                else:
                    final_node_val = value_node

                hurdle_mask = (S_node >= hurdle_H)
                option_values = np.where(hurdle_mask, final_node_val, prob_ficar * hold_values)
            else:
                option_values = prob_ficar * hold_values
        
        return option_values[0]
