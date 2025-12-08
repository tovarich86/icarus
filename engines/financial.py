"""
Módulo de Motores Financeiros (Engines).
Otimizado com Numba para alta performance em simulações iterativas.
"""
import numpy as np
import math
from numba import jit

# Constante para evitar divisão por zero
EPSILON = 1e-9

@jit(nopython=True, fastmath=True)
def _numba_norm_cdf(x):
    """Implementação da CDF Normal compatível com Numba (usando Error Function)."""
    return 0.5 * (1 + math.erf(x / 1.41421356))

@jit(nopython=True, fastmath=True)
def _calculate_lockup_discount_numba(volatility, lockup_time, stock_price, q):
    """Cálculo do desconto de iliquidez (Chaffe) otimizado."""
    if lockup_time <= EPSILON:
        return 0.0
    
    vol_sq_t = (volatility ** 2) * lockup_time
    # Expansão de Taylor ou verificação de segurança para números pequenos/grandes
    term_inner = 2 * (np.exp(vol_sq_t) - vol_sq_t - 1)
    
    if term_inner <= EPSILON: 
        return 0.0
    
    # Evita log de número negativo/zero
    val_log = np.exp(vol_sq_t) - 1
    if val_log <= EPSILON:
        return 0.0

    b = vol_sq_t + np.log(term_inner) - 2 * np.log(val_log)
    if b < 0: 
        return 0.0
    
    a = np.sqrt(b)
    # Usa a função interna compatível com Numba
    discount_val = stock_price * np.exp(-q * lockup_time) * (_numba_norm_cdf(a / 2) - _numba_norm_cdf(-a / 2))
    return discount_val

class FinancialMath:
    """
    Coleção de métodos estáticos para cálculos de engenharia financeira.
    """

    @staticmethod
    def bs_call(S, K, T, r, sigma, q=0.0):
        """
        Calcula Black-Scholes com tratamento robusto de bordas (Python puro/Numpy).
        """
        # 1. Tratamento de Tempo
        if T <= EPSILON:
            return max(S - K, 0.0)
        
        # 2. Tratamento de Volatilidade Zero (Intrínseco Descontado)
        if sigma <= EPSILON:
            val = (S * np.exp(-q * T)) - (K * np.exp(-r * T))
            return max(val, 0.0)
            
        # 3. Tratamento de Strike Zero (RSU)
        if K <= EPSILON:
            return S * np.exp(-q * T)
            
        # 4. Tratamento de Preço Spot Zero
        if S <= EPSILON:
            return 0.0

        try:
            sqrt_T = np.sqrt(T)
            d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrt_T)
            d2 = d1 - sigma * sqrt_T
            
            # Usa scipy apenas aqui se não estiver usando numba total, 
            # mas para consistência poderíamos usar _numba_norm_cdf se convertêssemos tudo.
            # Mantendo numpy padrão para compatibilidade com código legado não-compilado.
            from scipy.stats import norm
            return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        except Exception:
            return max(S - K, 0.0)

    @staticmethod
    @jit(nopython=True, fastmath=True)
    def binomial_custom_optimized(
        S, K, r, vol, q, 
        vesting_years, turnover_w, multiple_M, hurdle_H, 
        T_years, inflacao_anual, lockup_years, tipo_exercicio=0
    ):
        """
        Modelo Binomial Vetorizado e Compilado (Numba).
        Completa a lógica de indução retroativa (Backward Induction).
        """
        # --- Configuração da Grade ---
        # Garante steps mínimos e máximos para estabilidade
        total_steps = int(T_years * 252)
        if total_steps > 2500: total_steps = 2500
        if total_steps < 50: total_steps = 50 # Aumentei o mínimo para precisão
        
        dt = T_years / total_steps
        
        # Índice de Vesting
        Nv = int(vesting_years / dt)
        
        # Parâmetros CRR (Cox-Ross-Rubinstein)
        u = np.exp(vol * np.sqrt(dt))
        d = 1.0 / u
        p = (np.exp((r - q) * dt) - d) / (u - d)
        
        # Probabilidades de Turnover
        # Se total_steps > Nv, diluímos a probabilidade de saída nos passos pós-vesting
        # Se turnover_w é taxa anual (hazard rate):
        prob_ficar = np.exp(-turnover_w * dt)
        prob_sair = 1.0 - prob_ficar

        # --- 1. Estado Final (Vencimento) ---
        # Vetorização: Cria todos os nós finais de uma vez
        j_idx = np.arange(total_steps + 1)
        # S_T = S * u^(N-j) * d^j
        ST = S * (u ** (total_steps - j_idx)) * (d ** j_idx)
        
        # Strike Ajustado pela inflação até o final
        K_final = K * ((1 + inflacao_anual)**T_years)
        
        # Payoff no vencimento
        vals_base = ST.copy()
        
        # Aplica Lock-up no vencimento se necessário
        if lockup_years > 0:
            # Loop manual para Numba (ou vetorizado se a função suportar)
            for i in range(len(vals_base)):
                disc = _calculate_lockup_discount_numba(vol, lockup_years, vals_base[i], q)
                vals_base[i] -= disc
        
        # Payoff Básico: Max(S - K, 0)
        payoffs = np.maximum(vals_base - K_final, 0.0)
        
        # Aplica Barreira (Hurdle) no vencimento
        option_values = np.where(ST >= hurdle_H, payoffs, 0.0)
        
        # --- 2. Indução Retroativa (Backward Induction) ---
        # Itera do penúltimo passo até 0
        for i in range(total_steps - 1, -1, -1):
            # Recalcula Strike para o tempo i (Inflação)
            time_elapsed = i * dt
            K_curr = K * ((1 + inflacao_anual)**time_elapsed)
            
            # Valor de "Esperar" (Hold Value) - Valor presente esperado dos nós futuros
            # option_values tem tamanho i+2, queremos tamanho i+1
            # Vetorização: [0...N-1] e [1...N]
            hold_values = np.exp(-r * dt) * (p * option_values[:-1] + (1 - p) * option_values[1:])
            
            # Preço do Ativo neste nó (reconstrução otimizada)
            # S_node = S * u^(i-j) * d^j
            j_grid = np.arange(i + 1)
            S_node = S * (u ** (i - j_grid)) * (d ** j_grid)
            
            # Lógica Pós-Vesting (Exercício permitido dependendo do tipo)
            if i >= Nv:
                # Valor se exercer agora (considerando Lockup)
                S_exerc = S_node.copy()
                if lockup_years > 0:
                    for k in range(len(S_exerc)):
                        S_exerc[k] -= _calculate_lockup_discount_numba(vol, lockup_years, S_exerc[k], q)
                
                intrinsic_value = np.maximum(S_exerc - K_curr, 0.0)
                
                # Regra de Exercício Antecipado Subótimo (Múltiplo M)
                # Se S >= M * K, o empregado exerce (Force Exercise)
                force_exercise_mask = (S_node >= (multiple_M * K_curr)) & (tipo_exercicio == 0)
                
                # Valor considerando Turnover
                # Se sair (prob_sair), assume-se exercício imediato (se ITM) ou perda (se OTM/Bad Leaver).
                # Aqui assumimos simplificação: Saída = Exercício Antecipado
                value_node = (prob_sair * intrinsic_value) + (prob_ficar * hold_values)
                
                # Se forçado pelo Múltiplo M, prevalece o exercício
                # Se Americano (tipo=0) e racionalmente melhor exercer, exerce (max)
                if tipo_exercicio == 0:
                    rational_exercise = np.maximum(value_node, intrinsic_value)
                    final_node_val = np.where(force_exercise_mask, intrinsic_value, rational_exercise)
                else:
                    # Europeu
                    final_node_val = value_node

                # Aplica Hurdle (Barreira de Performance)
                # Se S < Hurdle, opção não vale nada ou apenas expectativa (hold)? 
                # Geralmente se S < Hurdle no vesting, perde. Se for só no exercício, checa aqui.
                # Assumindo Hurdle de Vesting:
                hurdle_mask = (S_node >= hurdle_H)
                option_values = np.where(hurdle_mask, final_node_val, prob_ficar * hold_values)
            
            else:
                # Pré-Vesting: Não pode exercer. Apenas desconta pelo risco de turnover.
                # Se sair antes do vesting, valor = 0 (Forfeiture).
                option_values = prob_ficar * hold_values
        
        return option_values[0]
