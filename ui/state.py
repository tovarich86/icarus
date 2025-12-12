import streamlit as st
from typing import List, Optional, Dict, Any
from core.domain import PlanAnalysisResult, Tranche, PricingModelType, SettlementType

class AppState:
    """
    Gerenciador centralizado do Session State do Streamlit (ViewModel).
    Garante tipagem forte e inicialização consistente dos dados.
    """
    
    # Chaves constantes para evitar erros de digitação ('magic strings')
    KEY_ANALYSIS = 'analysis_result'
    KEY_TRANCHES = 'tranches'
    KEY_CONTEXT = 'full_context_text'
    KEY_MC_CODE = 'mc_code'
    KEY_CALC_RESULTS = 'last_calc_results'

    @staticmethod
    def initialize():
        """Inicializa as variáveis de estado com valores padrão seguros."""
        if AppState.KEY_ANALYSIS not in st.session_state:
            st.session_state[AppState.KEY_ANALYSIS] = None
        
        if AppState.KEY_TRANCHES not in st.session_state:
            st.session_state[AppState.KEY_TRANCHES] = []
            
        if AppState.KEY_CONTEXT not in st.session_state:
            st.session_state[AppState.KEY_CONTEXT] = ""
            
        if AppState.KEY_MC_CODE not in st.session_state:
            st.session_state[AppState.KEY_MC_CODE] = ""
            
        if AppState.KEY_CALC_RESULTS not in st.session_state:
            st.session_state[AppState.KEY_CALC_RESULTS] = []

    # --- Getters e Setters Tipados ---

    @staticmethod
    def get_analysis() -> Optional[PlanAnalysisResult]:
        """Retorna o objeto de análise atual ou None."""
        return st.session_state.get(AppState.KEY_ANALYSIS)

    @staticmethod
    def set_analysis(result: PlanAnalysisResult):
        """Define a análise e sincroniza as tranches editáveis."""
        st.session_state[AppState.KEY_ANALYSIS] = result
        
        # Ao carregar uma nova análise, atualizamos a lista de tranches editáveis na UI
        if result and result.tranches:
             # Usamos model_copy() do Pydantic para desvincular da referência original
             st.session_state[AppState.KEY_TRANCHES] = [t.model_copy() for t in result.tranches]
        else:
             # Fallback: Cria uma tranche padrão se a lista vier vazia
             st.session_state[AppState.KEY_TRANCHES] = [
                 Tranche(vesting_date=1.0, proportion=1.0, expiration_date=5.0)
             ]

    @staticmethod
    def get_tranches() -> List[Tranche]:
        return st.session_state.get(AppState.KEY_TRANCHES, [])

    @staticmethod
    def set_tranches(tranches: List[Tranche]):
        st.session_state[AppState.KEY_TRANCHES] = tranches

    @staticmethod
    def get_context_text() -> str:
        return st.session_state.get(AppState.KEY_CONTEXT, "")
    
    @staticmethod
    def set_context_text(text: str):
        st.session_state[AppState.KEY_CONTEXT] = text

    @staticmethod
    def get_calc_results() -> List[Dict[str, Any]]:
        return st.session_state.get(AppState.KEY_CALC_RESULTS, [])

    @staticmethod
    def set_calc_results(results: List[Dict[str, Any]]):
        st.session_state[AppState.KEY_CALC_RESULTS] = results
    
    @staticmethod
    def get_mc_code() -> str:
        return st.session_state.get(AppState.KEY_MC_CODE, "")

    @staticmethod
    def set_mc_code(code: str):
        st.session_state[AppState.KEY_MC_CODE] = code

    # --- Ações de UI (Lógica de Negócio da Interface) ---

    @staticmethod
    def add_tranche_action():
        """Adiciona uma nova tranche vazia à lista."""
        current = AppState.get_tranches()
        # Pega a data da última tranche ou 0.0 se estiver vazia
        last_vesting = current[-1].vesting_date if current else 0.0
        
        new_tranche = Tranche(
            vesting_date=last_vesting + 1.0, 
            proportion=0.0, 
            expiration_date=10.0
        )
        current.append(new_tranche)
        AppState.set_tranches(current)

    @staticmethod
    def remove_last_tranche_action():
        """Remove a última tranche da lista."""
        current = AppState.get_tranches()
        if current:
            current.pop()
            AppState.set_tranches(current)

    @staticmethod
    def enable_manual_mode():
        """Cria um estado dummy para entrada manual."""
        manual_analysis = PlanAnalysisResult(
            summary="Modo Manual: Parâmetros definidos pelo usuário.",
            program_summary="Os dados do programa foram inseridos manualmente.",
            valuation_params="**Modo Manual**\n\n* Defina o Spot, Strike e Volatilidade nas caixas.",
            contract_features="Entrada Manual",
            methodology_rationale="Seleção manual do usuário.",
            model_recommended=PricingModelType.BLACK_SCHOLES_GRADED,
            settlement_type=SettlementType.EQUITY_SETTLED,
            tranches=[], # Será preenchido pelo set_analysis
            option_life_years=5.0,
            strike_price=10.0
        )
        
        AppState.set_analysis(manual_analysis)
        AppState.set_context_text("Modo Manual - Sem texto de contrato.")
