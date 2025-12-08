"""
Serviço de Estratégia e Seleção de Modelos.

Este módulo contém a lógica de negócios ("Regras do Jogo") para determinar
qual motor matemático é o mais adequado para um dado conjunto de características
de um plano de opções.

ATUALIZAÇÃO: Implementa lógica de correção para evitar sugestões de Monte Carlo
em casos onde modelos determinísticos (RSU) são suficientes.
"""

from core.domain import PlanAnalysisResult, PricingModelType

class ModelSelectorService:
    """
    Serviço responsável por auditar as características do plano e definir o modelo de precificação.
    """

    @staticmethod
    def select_model(analysis: PlanAnalysisResult) -> PlanAnalysisResult:
        """
        Analisa as características extraídas (flags) e determina o modelo matemático recomendado.

        Lógica de Decisão:
        1. Gatilhos de Mercado (TSR) -> Monte Carlo (Obrigatório).
        2. Strike Zero/Matching Shares -> RSU (Obrigatório).
           - CORREÇÃO: Se a IA sugerir Monte Carlo por causa de vesting/turnover, forçamos RSU.
        3. Características Americanas (Lock-up, Correção de Strike, Gap Longo) -> Binomial.
        4. Caso Padrão -> Black-Scholes Graded.

        Args:
            analysis (PlanAnalysisResult): O objeto contendo os dados extraídos da análise preliminar.

        Returns:
            PlanAnalysisResult: O mesmo objeto, enriquecido com a recomendação do modelo e racional ajustado.
        """
        
        # ---------------------------------------------------------------------
        # 1. Condições de Mercado (Path Dependent) exigem Simulação
        # ---------------------------------------------------------------------
        if analysis.has_market_condition:
            analysis.model_recommended = PricingModelType.MONTE_CARLO
            
            # Se a IA não forneceu justificativa, inserimos uma padrão
            if not analysis.methodology_rationale:
                analysis.methodology_rationale = (
                    "A presença de condições de mercado (ex: TSR, barreiras de preço) introduz dependência "
                    "da trajetória do preço, inviabilizando soluções fechadas. O Monte Carlo é estritamente "
                    "necessário para simular múltiplos cenários estocásticos para satisfazer essa condição."
                )
            return analysis

        # ---------------------------------------------------------------------
        # 2. Strike Zero ou Irrisório (Matching Shares / Restricted Stock)
        # ---------------------------------------------------------------------
        if analysis.strike_is_zero:
            analysis.model_recommended = PricingModelType.RSU
            
            # CORREÇÃO DE RACIONAL:
            # Frequentemente, LLMs sugerem Monte Carlo para RSUs com vesting acelerado ou turnover.
            # Aqui detectamos isso e corrigimos o texto para explicar a escolha pelo modelo RSU (Parcimônia).
            
            rationale_upper = analysis.methodology_rationale.upper() if analysis.methodology_rationale else ""
            
            if "MONTE CARLO" in rationale_upper and not analysis.has_market_condition:
                analysis.methodology_rationale = (
                    "Recomendação ajustada para RSU (Ações Restritas). Embora o contrato preveja cláusulas de "
                    "aceleração, turnover ou vesting escalonado, a ausência de condições de mercado (ex: TSR) "
                    "torna o uso de Monte Carlo desnecessariamente complexo. A metodologia padrão (IFRS 2) "
                    "para Strike Zero é o valor da ação à vista, descontado de dividendos não recebidos no "
                    "período de carência (Vestings tratados individualmente)."
                )
            elif not analysis.methodology_rationale:
                analysis.methodology_rationale = (
                    "O plano concede ações gratuitas ou com strike simbólico. A opcionalidade é irrelevante. "
                    "O modelo indicado é tratar como RSU (Valor à vista descontado de dividendos), "
                    "aplicando desconto de iliquidez (Chaffe) se houver Lock-up."
                )
            
            return analysis

        # ---------------------------------------------------------------------
        # 3. Características Exóticas / Americanas (Binomial)
        # ---------------------------------------------------------------------
        # Calcula gap entre vida da opção e vesting médio para identificar "Americanidade"
        gap_exercicio = analysis.option_life_years - analysis.get_avg_vesting()
        
        complex_features = (
            analysis.has_strike_correction or 
            gap_exercicio > 2.0 or 
            analysis.lockup_years > 0
        )

        if complex_features:
            analysis.model_recommended = PricingModelType.BINOMIAL
            
            if not analysis.methodology_rationale:
                analysis.methodology_rationale = (
                    "O plano possui características de exercício 'Americano' e barreiras complexas "
                    "(Lock-up ou Correção de Strike). O modelo Binomial (Lattice) discretiza o tempo, "
                    "permitindo capturar a otimalidade do exercício e o desconto de iliquidez nó a nó, "
                    "superando as limitações do Black-Scholes para estes casos."
                )
            return analysis

        # ---------------------------------------------------------------------
        # 4. Fallback: Black-Scholes Graded (Vanilla otimizado)
        # ---------------------------------------------------------------------
        analysis.model_recommended = PricingModelType.BLACK_SCHOLES_GRADED
        
        if not analysis.methodology_rationale:
            analysis.methodology_rationale = (
                "O plano não apresenta gatilhos de mercado ou barreiras complexas. "
                "O Black-Scholes-Merton (Graded) é o padrão de mercado eficiente, "
                "calculando cada tranche de vesting individualmente como uma opção europeia."
            )
        
        return analysis
