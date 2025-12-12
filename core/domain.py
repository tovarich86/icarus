"""
Módulo de Domínio (Core) - AI-First Version.

Define as estruturas de dados fundamentais utilizando Pydantic.
Isso permite validação robusta e geração automática de schemas JSON
para integração estruturada com LLMs (Gemini/OpenAI).
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, computed_field

class PricingModelType(str, Enum):
    """
    Enumeração dos modelos de precificação.
    Herda de str para facilitar a serialização JSON direta.
    """
    MONTE_CARLO = "Monte Carlo (Simulação)"
    BINOMIAL = "Binomial (Lattice Customizado)"
    BLACK_SCHOLES_GRADED = "Black-Scholes (Graded Vesting)"
    RSU = "RSU / Ações Restritas (Matching Shares)"
    UNDEFINED = "Indefinido"

class SettlementType(str, Enum):
    """
    Classificação Contábil do Instrumento (IFRS 2).
    """
    EQUITY_SETTLED = "Equity-Settled (Liquidação em Ações)"
    CASH_SETTLED = "Cash-Settled (Liquidação em Caixa/SARs)"
    HYBRID = "Híbrido (Opção de Liquidação)"
    UNDEFINED = "Indefinido"

class Tranche(BaseModel):
    """
    Representa uma tranche individual de vesting.
    """
    vesting_date: float = Field(
        ..., 
        description="Anos até o direito ser adquirido (Carência/Vesting). Ex: 1.0, 2.5",
        ge=0.0
    )
    proportion: float = Field(
        ..., 
        description="Proporção do total outorgado que esta tranche representa (0.0 a 1.0).",
        ge=0.0, 
        le=1.0
    )
    expiration_date: Optional[float] = Field(
        None, 
        description="Anos até o vencimento contratual (Life). Se nulo, assume o prazo global."
    )
    custom_strike: Optional[float] = Field(
        None, 
        description="Strike específico desta tranche, se diferente do global."
    )

class PlanAnalysisResult(BaseModel):
    """
    Resultado da análise do plano. 
    Este objeto serve como contrato de saída para a IA Generativa.
    """
    # --- Metadados Qualitativos (Gerados pela IA) ---
    summary: str = Field(
        ..., 
        description="Resumo executivo curto sobre o plano e suas regras principais."
    )
    program_summary: str = Field(
        ..., 
        description="Descrição detalhada narrativa do programa para o laudo."
    )
    valuation_params: str = Field(
        ..., 
        description="Texto formatado (Markdown) listando os principais parâmetros extraídos (Vesting, Lockup, etc)."
    )
    contract_features: str = Field(
        ..., 
        description="Lista de cláusulas chave identificadas no contrato."
    )
    methodology_rationale: str = Field(
        ..., 
        description="Justificativa técnica robusta explicando por que o modelo escolhido é o adequado segundo o IFRS 2."
    )
    
    # --- Decisões Técnicas ---
    model_recommended: PricingModelType = Field(
        PricingModelType.BLACK_SCHOLES_GRADED,
        description="O modelo matemático recomendado baseada na complexidade do plano."
    )
    settlement_type: SettlementType = Field(
        SettlementType.EQUITY_SETTLED,
        description="A classificação contábil (Equity vs Liability/Passivo)."
    )
    model_reason: str = Field(
        "", description="Razão curta para exibição em UI."
    )
    model_comparison: str = Field(
        "", description="Comparativo breve entre modelos (ex: Por que não usar BS Vanilla?)."
    )
    pros: List[str] = Field(
        default_factory=list, description="Lista de prós do modelo escolhido."
    )
    cons: List[str] = Field(
        default_factory=list, description="Lista de contras/limitações do modelo escolhido."
    )
    
    # --- Parâmetros Quantitativos (Inputs de Cálculo) ---
    tranches: List[Tranche] = Field(
        default_factory=list,
        description="Lista de tranches de vesting detectadas."
    )
    
    has_market_condition: bool = Field(
        False, description="True se houver gatilhos de mercado (TSR, Cotação Alvo)."
    )
    has_strike_correction: bool = Field(
        False, description="True se o strike for corrigido por índices (IGPM/IPCA)."
    )
    
    # Defaults Numéricos com Validação
    option_life_years: float = Field(5.0, ge=0.0)
    strike_price: float = Field(0.0, ge=0.0)
    strike_is_zero: bool = Field(False)
    turnover_rate: float = Field(0.0, ge=0.0, le=1.0)
    early_exercise_multiple: float = Field(2.0, ge=1.0)
    lockup_years: float = Field(0.0, ge=0.0)

    # --- Helpers (Mantidos como métodos ou propriedades computadas) ---

    def get_avg_vesting(self) -> float:
        """Calcula o prazo médio ponderado de vesting."""
        if not self.tranches:
            return 3.0
        # Pydantic garante que os tipos estão certos, então podemos calcular direto
        total_prop = sum(t.proportion for t in self.tranches)
        if total_prop == 0: return 0.0
        return sum(t.vesting_date * t.proportion for t in self.tranches) / total_prop

    @computed_field
    def is_liability(self) -> bool:
        """Propriedade computada: Verifica se é Passivo (Exige remensuração)."""
        return self.settlement_type in [SettlementType.CASH_SETTLED, SettlementType.HYBRID]

    class Config:
        # Permite usar .value nos Enums automaticamente na serialização
        use_enum_values = True
