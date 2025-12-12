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
        # CORREÇÃO APLICADA: Removidos os parênteses de is_liability (property)
        if analysis.is_liability:
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
                    "A presença de condições de mercado (ex: TSR ou Barreira de Preço) exige **Simulação de Monte Carlo**. "
                    "Métodos analíticos fechados não conseguem capturar a dependência da trajetória (Path Dependence) necessária "
                    "para precificar este gatilho de performance."
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
                    f"Recomendação ajustada para **{term_used}**. Apesar das regras de vesting, "
                    "a ausência de gatilhos de mercado torna o Monte Carlo desnecessário. "
                    "O valuation deve seguir o modelo de Valor Intrínseco Descontado (RSU)."
                )
            elif not analysis.methodology_rationale:
                analysis.methodology_rationale = (
                    f"O plano concede **{term_used}** (Strike Zero). O modelo indicado é o de RSU "
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
            
            # Constrói a lista de fatores determinantes de forma limpa
            factors = []
            if analysis.has_strike_correction:
                factors.append("Correção monetária do Strike (Indexação)")
            if gap_exercicio > 0.5:
                factors.append(f"Janela de exercício Americana extensa (~{gap_exercicio:.1f} anos)")
            if analysis.lockup_years > 0:
                factors.append(f"Restrição de Lock-up pós-exercício ({analysis.lockup_years} anos)")
            if ia_suggests_binomial and not factors:
                factors.append("Recomendação baseada na interpretação semântica das cláusulas contratuais (IA)")

            factors_str = "; ".join(factors) + "."

            # Racional Extensivo e Educativo
            analysis.methodology_rationale = (
                f"O modelo **Binomial (Lattice Customizado)** foi selecionado como o mais adequado. "
                f"Fatores determinantes identificados: {factors_str}\n\n"
                f"**Justificativa Técnica:** Diferente do modelo Black-Scholes (que assume parâmetros constantes e exercício em data fixa), "
                f"o modelo Binomial é capaz de incorporar a indexação do preço de exercício e modelar matematicamente "
                f"o comportamento de exercício antecipado (Early Exercise) dentro da janela de vigência, "
                f"atendendo com maior precisão aos requisitos do CPC 10 / IFRS 2 para este perfil de plano."
            )
            return analysis

        # ---------------------------------------------------------------------
        # 4. Fallback: Black-Scholes Graded (Vanilla otimizado)
        # ---------------------------------------------------------------------
        analysis.model_recommended = PricingModelType.BLACK_SCHOLES_GRADED
        
        if not analysis.methodology_rationale:
            analysis.methodology_rationale = (
                "Estrutura padrão (Opção Europeia/Plain Vanilla) sem barreiras complexas ou janelas longas de exercício. "
                "O **Black-Scholes-Merton (Graded)** é o padrão de mercado mais eficiente, "
                "calculando cada tranche individualmente conforme as melhores práticas."
            )
        
        return analysis
