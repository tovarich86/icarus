"""
Módulo de Domínio (Core).

Este módulo define as estruturas de dados fundamentais e enumeradores utilizados
em todo o sistema Icarus. Ele atua como a fonte única da verdade para os tipos
de dados, garantindo consistência entre a camada de Interface, Serviços e Engines Matemáticas.

Não deve conter lógica de negócios complexa ou dependências de bibliotecas externas
pesadas (como Streamlit ou Google Generative AI).
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List

class PricingModelType(Enum):
    """
    Enumeração dos modelos de precificação de opções suportados pelo sistema.
    
    Attributes:
        MONTE_CARLO: Simulação estocástica, ideal para path-dependency (barreiras, performance).
        BINOMIAL: Modelo Lattice (árvore), ideal para exercício antecipado (Americanas) e lock-up.
        BLACK_SCHOLES_GRADED: Modelo fechado padrão, aplicado por tranche individual.
        RSU: Restricted Stock Units / Matching Shares (Strike Zero ou irrelevante).
        UNDEFINED: Estado inicial ou erro de identificação.
    """
    MONTE_CARLO = "Monte Carlo (Simulação)"
    BINOMIAL = "Binomial (Lattice Customizado)"
    BLACK_SCHOLES_GRADED = "Black-Scholes (Graded Vesting)"
    RSU = "RSU / Ações Restritas (Matching Shares)"
    UNDEFINED = "Indefinido"

@dataclass
class Tranche:
    """
    Representa uma tranche individual de um plano de vesting escalonado (Graded Vesting).

    Attributes:
        vesting_date (float): O tempo até o vesting em anos (ex: 1.0, 2.5).
        proportion (float): A proporção desta tranche em relação ao grant total (0.0 a 1.0).
                            Ex: 0.25 representa 25% do total de opções outorgadas.
    """
    vesting_date: float
    proportion: float

@dataclass
class PlanAnalysisResult:
    """
    Agregador de resultados da análise qualitativa e quantitativa de um plano de opções.
    
    Esta estrutura armazena tanto a interpretação textual feita pela IA quanto os
    parâmetros numéricos extraídos para alimentação dos modelos matemáticos.

    Attributes:
        summary (str): Resumo executivo do plano.
        program_summary: str # Resumo "Jurídico/RH" (Participante, quantidade, regras gerais)
        valuation_params: str # Resumo "Quantitativo" (Carência, Dividendos, Lockup, Liquidação)
        methodology_rationale (str): Justificativa técnica profunda para a escolha do modelo.
        model_recommended (PricingModelType): O modelo matemático sugerido pela análise.
        model_reason (str): Justificativa curta/sintética (para UI).
        model_comparison (str): Texto comparativo entre o modelo escolhido e alternativas.
        pros (List[str]): Lista de pontos fortes da metodologia escolhida.
        cons (List[str]): Lista de limitações ou complexidades da metodologia.
        tranches (List[Tranche]): Cronograma de vesting identificado.
        has_market_condition (bool): Se True, indica gatilhos de mercado (ex: TSR) -> Requer Monte Carlo.
        has_strike_correction (bool): Se True, indica indexação do Strike (ex: IGPM/IPCA).
        option_life_years (float): Prazo contratual total da opção (Life).
        strike_price (float): Preço de exercício base.
        strike_is_zero (bool): Flag crítica. Se True, trata-se de RSU/Matching Shares.
        turnover_rate (float): Taxa anual esperada de saída de executivos (ex: 0.05 = 5%).
        early_exercise_multiple (float): Fator de exercício subótimo (M). Geralmente 2.0x o Strike.
        lockup_years (float): Período de restrição de venda pós-exercício (desconto de iliquidez).
    """
    # Metadados Qualitativos
    summary: str
    contract_features: str 
    methodology_rationale: str 
    model_recommended: PricingModelType
    model_reason: str
    model_comparison: str 
    pros: List[str]
    cons: List[str]
    
    # Parâmetros Quantitativos Extraídos
    tranches: List[Tranche] = field(default_factory=list)
    has_market_condition: bool = False 
    has_strike_correction: bool = False 
    option_life_years: float = 5.0 
    strike_price: float = 0.0
    strike_is_zero: bool = False
    turnover_rate: float = 0.0 
    early_exercise_multiple: float = 2.0 
    lockup_years: float = 0.0 
    
    def get_avg_vesting(self) -> float:
        """
        Calcula o prazo médio ponderado de vesting (Weighted Average Vesting Period).
        
        Utilizado para estimativas rápidas de gap de exercício. Caso não haja tranches
        definidas, retorna um valor padrão de 3.0 anos.

        Returns:
            float: A média ponderada dos anos de vesting.
        """
        if not self.tranches:
            return 3.0
        return sum(t.vesting_date * t.proportion for t in self.tranches)
