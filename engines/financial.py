"""
Módulo de Motores Financeiros (Engines) - IFRS 2 & Atuária
Versão Corrigida: Juros Contínuos, Proteção M e Estabilidade CRR.
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
    vol_sq_t = (volatility ** 2) * lockup_time
    term_inner = 2 * (np.exp(vol_sq_t) - vol_sq_t - 1)
    if term_inner <= EPSILON: return 0.0
    val_log = np.exp(vol_sq_t) - 1
    if val_log <= EPSILON: return 0.0
    b = vol_sq_t + np.log(term_inner) - 2 * np.log(val_log)
    if b < 0: return 0.0
    a = np.sqrt(b)
    return stock_price * np.exp(-q * lockup_time) * (_numba_norm_cdf(a/2) - _numba_norm_cdf(-a/2))

class FinancialMath:
    @staticmethod
    def calculate_lockup_discount(volatility, lockup_time, stock_price, q=0.0):
        return _calculate_lockup_discount_numba(float(volatility), float(lockup_time), float(stock_price), float(q))

    @staticmethod
    def bs_call(S, K, T, r_effective, sigma, q=0.0):
        S, K, T, r_eff, sigma, q = float(S), float(K), float(T), float(r_effective), float(sigma), float(q)
        
        # 1. Correção Matemática: Converter Taxa Efetiva (DI) para Contínua (Log-Return)
        # Se r_eff for muito pequeno, ln(1+r) ~ r, mas para 13% faz diferença.
        r = np.log(1 + r_eff)

        if T <= EPSILON: return max(S - K, 0.0)
        if sigma <= EPSILON:
            val = (S * np.exp(-q * T)) - (K * np.exp(-r * T))
            return max(val, 0.0)
        if K <= EPSILON: return S * np.exp(-q * T)
        if S <= EPSILON: return 0.0

        sqrt_T = np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def binomial_custom_optimized(
        S, K, r_effective, vol, q, 
        vesting_years, turnover_w, multiple_M, hurdle_H, 
        T_years, inflacao_anual, lockup_years, tipo_exercicio=0
    ):
        # 1. Correção da Taxa para Contínua
        r = np.log(1 + r_effective)

        # 2. Proteção de Volatilidade
        if vol < 1e-5:
            K_adj = K * ((1 + inflacao_anual)**T_years)
            return max((S * np.exp(-q * T_years)) - (K_adj * np.exp(-r * T_years)), 0.0)

        if T_years <= 1e-5: return max(S - K, 0.0)

        # Configuração CRR
        total_steps = int(T_years * 252)
        if total_steps > 2000: total_steps = 2000
        if total_steps < 50: total_steps = 50
        
        dt = T_years / total_steps
        Nv = int(vesting_years / dt)
        
        u = np.exp(vol * np.sqrt(dt))
        d = 1.0 / u
        
        # 3. Cálculo e Trava de Probabilidade (Estabilidade)
        # Evita p < 0 ou p > 1 se vol for muito baixa em relação aos juros
        denom = u - d
        if denom == 0: denom = 1e-9
        
        p = (np.exp((r - q) * dt) - d) / denom
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
            
            # --- CORREÇÃO DO MÚLTIPLO M (Lógica do Usuário) ---
            # Só força exercício se M > 1.0. Se M for irracional (<1), ignora a lógica de barreira.
            # E garantimos que S > M * K. 
            if multiple_M >= 1.0:
                force_exercise_mask = (S_node >= (multiple_M * K_curr)) & (tipo_exercicio == 0)
            else:
                # Se M < 1 (input errado do usuário ou desativado), nunca força exercício subótimo
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
