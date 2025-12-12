import unittest
from datetime import date
from typing import List, Optional
from dataclasses import dataclass, field
from enum import Enum

# --- 1. MOCKS DO DOMÍNIO (Simula o Icarus Core) ---
# Copie isso ou importe de core.domain se estiver no mesmo ambiente
class PricingModelType(Enum):
    BLACK_SCHOLES_GRADED = "Black-Scholes (Graded)"
    BINOMIAL = "Binomial (Lattice)"
    MONTE_CARLO = "Monte Carlo"
    RSU = "RSU / Cotação"
    UNDEFINED = "Indefinido"

class SettlementType(Enum):
    EQUITY_SETTLED = "Equity"
    CASH_SETTLED = "Cash"

@dataclass
class Tranche:
    vesting_date: float
    proportion: float
    expiration_date: Optional[float] = None

@dataclass
class PlanAnalysisResult:
    model_recommended: PricingModelType
    settlement_type: SettlementType
    has_market_condition: bool = False
    has_strike_correction: bool = False
    early_exercise_multiple: float = 2.0
    turnover_rate: float = 0.0
    lockup_years: float = 0.0

# --- 2. IMPORTAÇÃO DO SERVIÇO ---
# Ajuste o import conforme sua estrutura de pastas real
try:
    from services.report_service import ReportService
except ImportError:
    # Fallback: Se não conseguir importar, avisa o usuário
    print("ERRO: Não foi possível importar ReportService. Verifique o caminho.")
    ReportService = None

# --- 3. DADOS DE TESTE (CENÁRIOS) ---
# Estes dados replicam exatamente o que entra na tela do Streamlit
CENARIOS_TESTE = [
    {
        "id": "CENARIO_1_SOP_PRIVADA",
        "inputs_manual": {
            "empresa": {"nome": "Tech Unicorn Ltda", "ticker": "", "capital_aberto": False},
            "programa": {
                "nome": "Plano de Opções 2024", "data_outorga": date(2024, 1, 1), 
                "qtd_beneficiarios": 12, "metodologia": "BLACK_SCHOLES", "forma_liquidacao": "ACOES",
                "tipo_detalhado": "Stock Option Plan"
            },
            "responsavel": {"nome": "Auditor 1", "cargo": "Sócio", "email": "auditor@email.com"},
            "contab": {"taxa_turnover": 0.05, "tem_encargos": False},
            "calculo_extra": {"metodo_privado": "Fluxo de Caixa Descontado", "indice_correcao_nome": "IGP-M"}
        },
        # Estado simulado da Análise (IA + Strategy)
        "analysis_mock": PlanAnalysisResult(
            model_recommended=PricingModelType.BLACK_SCHOLES_GRADED,
            settlement_type=SettlementType.EQUITY_SETTLED,
            has_strike_correction=True, lockup_years=1.0
        ),
        # Resultados simulados do Cálculo Matemático
        "calc_results_mock": [
            {"TrancheID": 1, "S": 100.0, "K": 100.0, "Vol": 0.30, "r": 0.10, "T": 10.0, "q": 0.025, 
             "FV Unit": 25.00, "FV Ponderado": 25000.0}
        ],
        "tranches_mock": [Tranche(1.0, 1.0)],
        
        # O QUE ESPERAMOS NO LAUDO (Asserções)
        "expected": {
            "metodo_privado": "Fluxo de Caixa Descontado",
            "fv_unitario": "R$ 25,00",
            "modelo_txt": "BLACK_SCHOLES"
        }
    },
    {
        "id": "CENARIO_2_PSU_MONTECARLO",
        "inputs_manual": {
            "empresa": {"nome": "Mineração Global S.A.", "ticker": "VALE3", "capital_aberto": True},
            "programa": {
                "nome": "Programa 2024", "data_outorga": date(2024, 6, 30), 
                "qtd_beneficiarios": 5, "metodologia": "MONTE_CARLO", "forma_liquidacao": "CAIXA",
                "tipo_detalhado": "Performance Shares"
            },
            "responsavel": {"nome": "Quant", "cargo": "Analista", "email": "q@q.com"},
            "contab": {"taxa_turnover": 0.0, "tem_encargos": True},
            "calculo_extra": {}
        },
        "analysis_mock": PlanAnalysisResult(
            model_recommended=PricingModelType.MONTE_CARLO,
            settlement_type=SettlementType.CASH_SETTLED,
            has_market_condition=True
        ),
        "calc_results_mock": [
            {"TrancheID": 1, "S": 65.0, "K": 0.0, "Vol": 0.35, "r": 0.11, "T": 3.0, "q": 0.0, 
             "FV Unit": 55.00, "FV Ponderado": 1000000.0}
        ],
        "tranches_mock": [Tranche(3.0, 1.0)],
        
        "expected": {
            "modelo_txt": "MONTE_CARLO",
            "forma_liq": "CAIXA", # Deve acusar Passivo
            "texto_perf": "condições de mercado (TSR)"
        }
    }
]

# --- 4. CLASSE DE TESTE UNITÁRIO ---
class TestIcarusReportGeneration(unittest.TestCase):

    def setUp(self):
        if not ReportService:
            self.skipTest("ReportService não encontrado")

    def test_cenarios_consistencia(self):
        print("\n>>> INICIANDO TESTES DE CONSISTÊNCIA DE LAUDO <<<")
        
        for cenario in CENARIOS_TESTE:
            with self.subTest(cenario=cenario["id"]):
                print(f"Testing: {cenario['id']}...", end=" ")
                
                # 1. Executa a Geração do Contexto (O Core do ReportService)
                context = ReportService.generate_report_context(
                    analysis_result=cenario["analysis_mock"],
                    tranches=cenario["tranches_mock"],
                    calc_results=cenario["calc_results_mock"],
                    manual_inputs=cenario["inputs_manual"]
                )
                
                # 2. Asserções (Verifica se a lógica funcionou)
                
                # Checa Modelo
                self.assertEqual(context["programa"]["metodologia"], cenario["expected"].get("modelo_txt"), 
                                 f"Erro Modelo em {cenario['id']}")
                
                # Checa Fair Value Formatado
                if "fv_unitario" in cenario["expected"]:
                    fv_gerado = context["tabelas"]["resultados_fair_value"][0]["fv_final"]
                    self.assertEqual(fv_gerado, cenario["expected"]["fv_unitario"],
                                     "Erro na formatação do Fair Value")

                # Checa Campos Específicos (Privado vs Publico)
                if "metodo_privado" in cenario["expected"]:
                    self.assertEqual(context["calculo"]["metodo_precificacao_privado"], cenario["expected"]["metodo_privado"],
                                     "Erro no método de precificação privado")
                
                # Checa Lógica de Texto Condicional
                if "texto_perf" in cenario["expected"]:
                    self.assertIn(cenario["expected"]["texto_perf"], context["regras"]["texto_performance"],
                                  "Erro na descrição da regra de performance")

                print("OK ✅")

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
