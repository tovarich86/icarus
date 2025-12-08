"""
Módulo de Domínio (Core).

Define as estruturas de dados fundamentais e enumeradores utilizados em todo o sistema Icarus.
Focado em conformidade com IFRS 2 (CPC 10) para remensuração e classificação correta.
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional

class PricingModelType(Enum):
    """
    Enumeração dos modelos de precificação de opções suportados.
    """
    MONTE_CARLO = "Monte Carlo (Simulação)"
    BINOMIAL = "Binomial (Lattice Customizado)"
    BLACK_SCHOLES_GRADED = "Black-Scholes (Graded Vesting)"
    RSU = "RSU / Ações Restritas (Matching Shares)"
    UNDEFINED = "Indefinido"

class SettlementType(Enum):
    """
    Classificação Contábil do Instrumento (IFRS 2).
    Define se o plano cria uma Reserva de Capital (Equity) ou um Passivo (Cash).
    """
    EQUITY_SETTLED = "Equity-Settled (Liquidação em Ações)"
    CASH_SETTLED = "Cash-Settled (Liquidação em Caixa/SARs)"
    HYBRID = "Híbrido (Opção de Liquidação)"
    UNDEFINED = "Indefinido"

@dataclass
class Tranche:
    """
    Representa uma tranche individual de vesting.
    Permite distinguir o período de aquisição (Vesting) do prazo contratual total (Expiration).
    
    Attributes:
        vesting_date (float): Anos até o direito ser adquirido (Carência).
        proportion (float): % do total outorgado (0.0 a 1.0).
        expiration_date (float): Anos até o vencimento do direito (Maturity/Life).
                                 Geralmente > vesting_date.
                                 Se None, assume-se igual ao option_life global.
        custom_strike (Optional[float]): Strike específico desta tranche (se houver).
    """
    vesting_date: float
    proportion: float
    expiration_date: Optional[float] = None
    custom_strike: Optional[float] = None

@dataclass
class PlanAnalysisResult:
    """
    Agregador de resultados da análise qualitativa e quantitativa.
    Agora suporta classificação contábil para remensuração.
    """
    # Metadados Qualitativos
    summary: str
    program_summary: str       
    valuation_params: str
    contract_features: str 
    methodology_rationale: str 
    model_recommended: PricingModelType
    settlement_type: SettlementType  # NOVO: Define necessidade de remensuração
    model_reason: str
    model_comparison: str 
    pros: List[str]
    cons: List[str]
    
    # Parâmetros Quantitativos Extraídos
    tranches: List[Tranche] = field(default_factory=list)
    has_market_condition: bool = False 
    has_strike_correction: bool = False 
    
    # Parâmetros Globais (Defaults)
    option_life_years: float = 5.0  # Prazo contratual padrão
    strike_price: float = 0.0
    strike_is_zero: bool = False
    turnover_rate: float = 0.0 
    early_exercise_multiple: float = 2.0 
    lockup_years: float = 0.0 
    
    def get_avg_vesting(self) -> float:
        """Calcula o prazo médio ponderado de vesting."""
        if not self.tranches:
            return 3.0
        return sum(t.vesting_date * t.proportion for t in self.tranches)

    def is_liability(self) -> bool:
        """Helper para verificar se é Passivo (Exige remensuração)."""
        return self.settlement_type in [SettlementType.CASH_SETTLED, SettlementType.HYBRID]
