"""
Módulo de Motores Financeiros (Engines).

Este módulo contém as implementações puras dos modelos matemáticos de precificação.
As funções aqui devem ser estáticas, stateless (sem estado interno) e altamente otimizadas
usando NumPy.

Modelos incluídos:
1. Black-Scholes-Merton (Closed Form).
2. Modelo Chaffe (Desconto de Iliquidez/Lock-up).
3. Binomial Lattice Customizado (Vetorizado).

Dependências:
- numpy: Para vetorização e performance.
- scipy.stats: Para funções de distribuição normal (norm.cdf).
"""

import numpy as np
from scipy.stats import norm

class FinancialMath:
    """
    Coleção de métodos estáticos para cálculos de engenharia financeira.
    Não deve ser instanciada.
    """

    @staticmethod
    def bs_call(
        S: float, 
        K: float, 
        T: float, 
        r: float, 
        sigma: float, 
        q: float = 0.0
    ) -> float:
        """
        Calcula o preço de uma opção de compra (Call) Europeia usando Black-Scholes-Merton.

        Lida com casos de borda (T=0, Vol=0, K=0) para evitar erros de divisão por zero.

        Args:
            S (float): Preço Spot do ativo subjacente.
            K (float): Preço de exercício (Strike).
            T (float): Tempo até o vencimento em anos.
            r (float): Taxa livre de risco (anual, contínua).
            sigma (float): Volatilidade do ativo (anual).
            q (float, optional): Dividend yield contínuo. Defaults to 0.0.

        Returns:
            float: O valor justo (Fair Value) da opção.
        """
        # 1. Tratamento de Tempo (Vencido ou muito curto)
        if T <= 1e-6:
            return max(S - K, 0.0)
        
        # 2. Tratamento de Volatilidade Zero ou Negativa (Valor Intrínseco Descontado)
        if sigma <= 1e-6:
            val = (S * np.exp(-q * T)) - (K * np.exp(-r * T))
            return max(val, 0.0)
            
        # 3. Tratamento de Strike Zero ou Negativo (RSU / Forward)
        if K <= 1e-6:
            return S * np.exp(-q * T)
            
        # 4. Tratamento de Preço Spot Zero
        if S <= 1e-6:
            return 0.0

        try:
            d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        except Exception:
            # Fallback seguro em caso de overflow numérico extremo
            return max(S - K, 0.0)

    @staticmethod
    def calculate_lockup_discount(
        volatility: float, 
        lockup_time: float, 
        stock_price: float, 
        q: float
    ) -> float:
        """
        Calcula o valor monetário do desconto por falta de liquidez (DLOM) 
        usando o modelo de Chaffe (Put Option Proxy).

        Args:
            volatility (float): Volatilidade do ativo.
            lockup_time (float): Tempo de restrição de venda em anos.
            stock_price (float): Preço do ativo no momento do cálculo.
            q (float): Dividend yield.

        Returns:
            float: O valor monetário a ser subtraído do preço do ativo.
        """
        if lockup_time <= 0: return 0.0
        
        vol_sq_t = (volatility ** 2) * lockup_time
        term_inner = 2 * (np.exp(vol_sq_t) - vol_sq_t - 1)
        
        if term_inner <= 0: return 0.0
        
        b = vol_sq_t + np.log(term_inner) - 2 * np.log(np.exp(vol_sq_t) - 1)
        if b < 0: return 0.0
        
        a = np.sqrt(b)
        discount_val = stock_price * np.exp(-q * lockup_time) * (norm.cdf(a / 2) - norm.cdf(-a / 2))
        
        return discount_val

    @staticmethod
    def binomial_custom_optimized(
        S: float, 
        K: float, 
        r: float, 
        vol: float, 
        q: float, 
        vesting_years: float, 
        turnover_w: float, 
        multiple_M: float, 
        hurdle_H: float, 
        T_years: float, 
        inflacao_anual: float, 
        lockup_years: float, 
        tipo_exercicio: int = 0
    ) -> float:
        """
        Modelo Binomial (Lattice) vetorizado para precificação de Opções de Empregados (ESOP).
        
        Suporta características exóticas:
        - Exercício Antecipado Subótimo (Fator M).
        - Taxa de Rotatividade (Turnover/Forfeiture).
        - Lock-up pós-exercício (Desconto dinâmico no nó).
        - Correção inflacionária do Strike.
        - Barreiras de preço (Hurdle).

        [Image of Binomial Options Pricing Model lattice tree diagram]

        Args:
            S (float): Spot Price inicial.
            K (float): Strike Price inicial.
            r (float): Taxa livre de risco.
            vol (float): Volatilidade.
            q (float): Dividend yield.
            vesting_years (float): Tempo até o vesting da tranche.
            turnover_w (float): Taxa de saída anual (hazard rate).
            multiple_M (float): Múltiplo de exercício antecipado (S >= M * K).
            hurdle_H (float): Barreira de preço para vesting (0 se não houver).
            T_years (float): Tempo total de vida da opção.
            inflacao_anual (float): Taxa de correção do Strike.
            lockup_years (float): Tempo de restrição de venda pós-exercício.
            tipo_exercicio (int, optional): 0 = American (com M), 1 = European. Defaults to 0.

        Returns:
            float: Valor presente da opção (nó inicial).
        """
        # Discretização Otimizada
        total_steps = int(T_years * 252) # Aproximadamente dias úteis
        
        # Limites de segurança para performance (evita travamento da memória)
        if total_steps > 2000: total_steps = 2000
        if total_steps < 10: total_steps = 10
        
        dt = T_years / total_steps
        
        # Índices de Vesting (Step no qual o vesting ocorre)
        Nv = int(vesting_years / dt)
        if Nv > total_steps: Nv = total_steps
        
        # Parâmetros CRR (Cox-Ross-Rubinstein)
        u = np.exp(vol * np.sqrt(dt))
        d = 1.0 / u
        p = (np.exp((r - q) * dt) - d) / (u - d)
        
        # Probabilidades de Turnover (Saída do funcionário)
        # Se total_steps > Nv, diluímos a probabilidade de saída nos passos pós-vesting
        if total_steps > Nv:
            dt_vesting = 1.0 / (total_steps - Nv)
        else:
            dt_vesting = 0.0
            
        # 1. Árvore de Preços no Vencimento (Passo N, vetorizado)
        # Cria um array de j=0 a N
        j_idx = np.arange(total_steps + 1)
        # S_T = S * u^(N-j) * d^j
        ST = S * (u ** (total_steps - j_idx)) * (d ** j_idx)
        
        # 2. Strike no Vencimento (Ajustado por Inflação composta)
        K_final = K * ((1 + inflacao_anual)**T_years)
        
        # 3. Valor da Opção no Vencimento (Payoff Boundary Condition)
        vals_base = ST.copy()
        
        # Aplica desconto de Lockup no vencimento se necessário
        if lockup_years > 0:
            discounts = FinancialMath.calculate_lockup_discount(vol, lockup_years, ST, q)
            vals_base -= discounts
            
        payoffs = np
