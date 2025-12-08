"""
Serviço de Estratégia e Seleção de Modelos.

Este módulo contém a lógica de negócios ("Regras do Jogo") para determinar
qual motor matemático é o mais adequado, considerando agora as nuances de
Contabilidade (Equity vs Liability) e a estrutura temporal (Vesting vs Life).
"""

from core.domain import PlanAnalysisResult, PricingModelType, SettlementType

class ModelSelectorService:
    """
    Serviço responsável por auditar as características do plano e definir o modelo de precificação.
    """

    @staticmethod
    def select_model(analysis: PlanAnalysisResult) -> PlanAnalysisResult:
        """
        Analisa as características extraídas e determina o modelo matemático recomendado.

        Novas Regras de Decisão:
        1. Classificação Contábil: Alerta sobre remensuração se for Cash-Settled.
        2. Gap de Exercício: Usa dados precisos das tranches para sugerir Binomial.
        """
        
        # ---------------------------------------------------------------------
        # 0. Enriquecimento de Racional (Contabilidade)
        # ---------------------------------------------------------------------
        # Se for passivo (Cash-Settled), garantimos que o racional mencione isso.
        if analysis.is_liability():
            warning_text = f" [ATENÇÃO: Plano {analysis.settlement_type.value}. Requer remensuração do Fair Value a cada data de balanço]."
            if warning_text not in analysis.methodology_rationale:
                analysis.methodology_rationale += warning_text

        # ---------------------------------------------------------------------
        # 1. Condições de Mercado (Path Dependent) -> Monte Carlo
        # ---------------------------------------------------------------------
        if analysis.has_market_condition:
            analysis.model_recommended = PricingModelType.MONTE_CARLO
            if not analysis.methodology_rationale:
                analysis.methodology_rationale = (
                    "A presença de condições de mercado (ex: TSR) exige Simulação de Monte Carlo "
                    "para capturar a dependência da trajetória do preço."
                )
            return analysis

        # ---------------------------------------------------------------------
        # 2. Strike Zero ou Irrisório -> RSU / Phantom Shares
        # ---------------------------------------------------------------------
        if analysis.strike_is_zero:
            analysis.model_recommended = PricingModelType.RSU
            
            # Refinamento do Racional baseado na Liquidação
            term_used = "Phantom Shares" if analysis.settlement_type == SettlementType.CASH_SETTLED else "Ações Restritas (RSU)"
            
            rationale_upper = analysis.methodology_rationale.upper() if analysis.methodology_rationale else ""
            
            # Evita sugerir Monte Carlo para RSU simples (erro comum de LLMs)
            if "MONTE CARLO" in rationale_upper and not analysis.has_market_condition:
                analysis.methodology_rationale = (
                    f"Recomendação ajustada para {term_used}. Apesar das regras de vesting, "
                    "a ausência de gatilhos de mercado torna o Monte Carlo desnecessário. "
                    "O valuation deve seguir o modelo de Valor Intrínseco Descontado (RSU)."
                )
            elif not analysis.methodology_rationale:
                analysis.methodology_rationale = (
                    f"O plano concede {term_used} (Strike Zero). O modelo indicado é o de RSU "
                    "(Valor à vista descontado de dividendos), aplicando desconto de iliquidez (Chaffe) se houver Lock-up."
                )
            
            return analysis

        # ---------------------------------------------------------------------
        # 3. Características Americanas / Barreiras -> Binomial
        # ---------------------------------------------------------------------
        # Cálculo mais preciso do "Gap de Exercício" (Janela de Oportunidade)
        # Se a opção vence muito tempo depois de vestir, o valor do exercício antecipado (Americano) é relevante.
        
        avg_vesting = analysis.get_avg_vesting()
        
        # Tenta calcular a média de vencimento das tranches, se disponível
        if analysis.tranches and analysis.tranches[0].expiration_date:
            avg_life = sum(t.expiration_date * t.proportion for t in analysis.tranches)
        else:
            avg_life = analysis.option_life_years

        gap_exercicio = avg_life - avg_vesting
        
        # Critérios para Binomial
        complex_features = (
            analysis.has_strike_correction or       # Strike indexado (IGPM, etc)
            gap_exercicio > 2.0 or                  # Janela de exercício longa (Valor de tempo relevante)
            analysis.lockup_years > 0               # Restrição de venda pós-exercício
        )

        if complex_features:
            analysis.model_recommended = PricingModelType.BINOMIAL
            
            if not analysis.methodology_rationale:
                analysis.methodology_rationale = (
                    f"O plano possui características 'Americanas' (Janela de exercício de ~{gap_exercicio:.1f} anos) "
                    "ou barreiras complexas (Lock-up/Correção). O modelo Binomial (Lattice) é superior ao "
                    "Black-Scholes pois captura a decisão ótima de exercício antecipado e descontos de iliquidez."
                )
            return analysis

        # ---------------------------------------------------------------------
        # 4. Fallback: Black-Scholes Graded (Vanilla otimizado)
        # ---------------------------------------------------------------------
        analysis.model_recommended = PricingModelType.BLACK_SCHOLES_GRADED
        
        if not analysis.methodology_rationale:
            analysis.methodology_rationale = (
                "O plano segue estrutura padrão (Opção Europeia/Plain Vanilla) sem gatilhos complexos. "
                "O Black-Scholes-Merton (Graded) é o padrão de mercado mais eficiente, "
                "calculando cada tranche individualmente."
            )
        
        return analysis
