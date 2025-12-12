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
        3. IA Authority: Respeita a sugestão inicial da IA se for Binomial.
        """
        
        # ---------------------------------------------------------------------
        # 0. Enriquecimento de Racional (Contabilidade)
        # ---------------------------------------------------------------------
        # Se for passivo (Cash-Settled), garantimos que o racional mencione isso.
        iif analysis.is_liability:  # <--- REMOVA OS PARÊNTESES
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
        
        avg_vesting = analysis.get_avg_vesting()
        
        # Tenta calcular a média de vencimento das tranches, se disponível
        if analysis.tranches and analysis.tranches[0].expiration_date:
            avg_life = sum(t.expiration_date * t.proportion for t in analysis.tranches)
        else:
            avg_life = analysis.option_life_years

        gap_exercicio = avg_life - avg_vesting
        
        # Verifica se a IA já recomendou Binomial explicitamente
        ia_suggests_binomial = (analysis.model_recommended == PricingModelType.BINOMIAL)
        
        # Critérios Combinados para Binomial
        should_use_binomial = (
            ia_suggests_binomial or                 # Respeita a IA (Prioridade)
            analysis.has_strike_correction or       # Strike indexado (IGPM, etc)
            gap_exercicio > 0.5 or                  # Janela de exercício longa (> 6 meses)
            analysis.lockup_years > 0               # Restrição de venda pós-exercício
        )

        if should_use_binomial:
            analysis.model_recommended = PricingModelType.BINOMIAL
            
            # Se o racional estiver vazio ou mencionar Black-Scholes incorretamente, atualiza
            if not analysis.methodology_rationale or "Black-Scholes" in analysis.methodology_rationale:
                analysis.methodology_rationale = (
                    f"Modelo Binomial selecionado. Fatores determinantes: "
                    f"{'Recomendação IA, ' if ia_suggests_binomial else ''}"
                    f"{'Janela de Exercício Americana (~{gap_exercicio:.1f} anos), ' if gap_exercicio > 0.5 else ''}"
                    f"{'Lock-up, ' if analysis.lockup_years > 0 else ''}"
                    f"{'Correção de Strike.' if analysis.has_strike_correction else ''}"
                ).strip(", ")
            return analysis

        # ---------------------------------------------------------------------
        # 4. Fallback: Black-Scholes Graded (Vanilla otimizado)
        # ---------------------------------------------------------------------
        analysis.model_recommended = PricingModelType.BLACK_SCHOLES_GRADED
        
        if not analysis.methodology_rationale:
            analysis.methodology_rationale = (
                "Estrutura padrão (Opção Europeia/Plain Vanilla) sem barreiras complexas ou janelas longas de exercício. "
                "O Black-Scholes-Merton (Graded) é o padrão de mercado mais eficiente, "
                "calculando cada tranche individualmente."
            )
        
        return analysis
