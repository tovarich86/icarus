import unittest
import sys
import os
from datetime import date
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

# --- 1. CONFIGURAÇÃO DE AMBIENTE ---
# Adiciona o diretório atual ao path para encontrar a pasta 'services'
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Tenta importar o serviço. Se falhar, avisa o usuário.
try:
    from services.report_service import ReportService
except ImportError as e:
    print(f"\n[ERRO CRÍTICO] Não foi possível importar 'services.report_service'. Detalhe: {e}")
    print("Certifique-se de estar na raiz do projeto e que as dependências (docxtpl) estão instaladas.\n")
    sys.exit(1)

# --- 2. MOCKS DO DOMÍNIO (Simulação do Core do Icarus) ---
class PricingModelType(Enum):
    BLACK_SCHOLES_GRADED = "Black-Scholes (Graded)"
    BINOMIAL = "Binomial (Lattice)"
    MONTE_CARLO = "Monte Carlo"
    RSU = "RSU / Cotação"
    COTACAO = "Cotação Direta" # Alias para RSU em alguns contextos
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

# --- 3. CENÁRIOS DE TESTE (Baseados nos seus Golden Masters) ---
CENARIOS_TESTE = [
    # CENÁRIO 1: SOP Privada (Black-Scholes)
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
            "calculo_extra": {
                "metodo_privado": "Fluxo de Caixa Descontado", 
                "indice_correcao_nome": "IGP-M",
                "moeda_selecionada": "BRL", "cenario_dividendos": "PENALIZA"
            }
        },
        "analysis_mock": PlanAnalysisResult(
            model_recommended=PricingModelType.BLACK_SCHOLES_GRADED,
            settlement_type=SettlementType.EQUITY_SETTLED,
            has_strike_correction=True, lockup_years=1.0
        ),
        "calc_results_mock": [
            {"TrancheID": 1, "S": 100.0, "K": 100.0, "Vol": 0.30, "r": 0.10, "T": 10.0, "q": 0.025, 
             "FV Unit": 25.00, "FV Ponderado": 25000.0}
        ],
        "tranches_mock": [Tranche(1.0, 1.0)],
        "expected": {
            "metodo_privado": "Fluxo de Caixa Descontado",
            "fv_unitario": "R$ 25,00",
            "modelo_txt": "BLACK_SCHOLES",
            "qtd_beneficiarios": 12 # Verifica se a qtd foi passada corretamente
        }
    },

    # CENÁRIO 2: PSU (Monte Carlo + Caixa)
    {
        "id": "CENARIO_2_PSU_MONTECARLO",
        "inputs_manual": {
            "empresa": {"nome": "Mineração Global S.A.", "ticker": "VALE3", "capital_aberto": True, "bolsa_nome": "B3"},
            "programa": {
                "nome": "Programa 2024", "data_outorga": date(2024, 6, 30), 
                "qtd_beneficiarios": 5, "metodologia": "MONTE_CARLO", "forma_liquidacao": "CAIXA",
                "tipo_detalhado": "Performance Shares"
            },
            "responsavel": {"nome": "Quant", "cargo": "Analista", "email": "q@q.com"},
            "contab": {"taxa_turnover": 0.0, "tem_encargos": True},
            "calculo_extra": {"moeda_selecionada": "BRL", "cenario_dividendos": "PAGO"}
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
            "forma_liq": "CAIXA", # Deve ser Caixa (Passivo)
            "texto_perf": "condições de mercado (TSR)",
            "fv_unitario": "R$ 55,00"
        }
    },

    # CENÁRIO 3: RSU Simples (Cotação)
    {
        "id": "CENARIO_3_RSU_SIMPLES",
        "inputs_manual": {
            "empresa": {"nome": "Varejo S.A.", "ticker": "LREN3", "capital_aberto": True, "bolsa_nome": "B3"},
            "programa": {
                "nome": "Plano RSU 2024", "data_outorga": date(2024, 3, 15), 
                "qtd_beneficiarios": 200, "metodologia": "COTACAO", "forma_liquidacao": "ACOES",
                "tipo_detalhado": "Restricted Stock Units"
            },
            "responsavel": {"nome": "RH", "cargo": "Gerente", "email": "rh@email.com"},
            "contab": {"taxa_turnover": 0.10, "tem_encargos": True, "tem_metas_nao_mercado": True},
            "calculo_extra": {"moeda_selecionada": "BRL", "cenario_dividendos": "PAGO"}
        },
        "analysis_mock": PlanAnalysisResult(
            model_recommended=PricingModelType.RSU,
            settlement_type=SettlementType.EQUITY_SETTLED,
            has_market_condition=False
        ),
        "calc_results_mock": [
            {"TrancheID": 1, "S": 15.00, "K": 0.0, "Vol": 0.0, "r": 0.10, "T": 1.0, "q": 0.0,
             "FV Unit": 15.00, "FV Ponderado": 15000.0}
        ],
        "tranches_mock": [Tranche(1.0, 1.0)],
        "expected": {
            "modelo_txt": "COTACAO", 
            "fv_unitario": "R$ 15,00",
            "qtd_beneficiarios": 200
        }
    },

    # CENÁRIO 4: SOP Binomial (Americano)
    {
        "id": "CENARIO_4_SOP_BINOMIAL",
        "inputs_manual": {
            "empresa": {"nome": "Tech Complex Ltda", "ticker": "", "capital_aberto": False},
            "programa": {
                "nome": "SOP Binomial", "data_outorga": date(2024, 1, 1), 
                "qtd_beneficiarios": 3, "metodologia": "BINOMIAL", "forma_liquidacao": "ACOES",
                "tipo_detalhado": "Opções Americanas"
            },
            "responsavel": {"nome": "Matemático", "cargo": "Consultor", "email": "math@email.com"},
            "contab": {"taxa_turnover": 0.0, "tem_encargos": False},
            "calculo_extra": {
                "metodo_privado": "Múltiplos",
                "moeda_selecionada": "USD", # Testa Bond Americano
                "cenario_dividendos": "ZERO"
            }
        },
        "analysis_mock": PlanAnalysisResult(
            model_recommended=PricingModelType.BINOMIAL,
            settlement_type=SettlementType.EQUITY_SETTLED,
            early_exercise_multiple=2.5
        ),
        "calc_results_mock": [
            {"TrancheID": 1, "S": 50.0, "K": 50.0, "Vol": 0.40, "r": 0.105, "T": 10.0, "q": 0.0,
             "FV Unit": 12.00, "FV Ponderado": 12000.0}
        ],
        "tranches_mock": [Tranche(4.0, 1.0)],
        "expected": {
            "modelo_txt": "BINOMIAL", 
            "metodo_privado": "Múltiplos",
            "moeda_ref": "USD" # Deve validar se a taxa de juros foi para T-Bond
        }
    }
]

# --- 4. CLASSE DE TESTE UNITÁRIO ---
class TestIcarusReportGeneration(unittest.TestCase):

    def test_cenarios_consistencia(self):
        print("\n>>> INICIANDO TESTES DE CONSISTÊNCIA DE LAUDO (4 CENÁRIOS) <<<")
        
        for cenario in CENARIOS_TESTE:
            with self.subTest(cenario=cenario["id"]):
                print(f"Testing: {cenario['id']}...", end=" ")
                
                # 1. Execução: Gera o Contexto do Relatório
                context = ReportService.generate_report_context(
                    analysis_result=cenario["analysis_mock"],
                    tranches=cenario["tranches_mock"],
                    calc_results=cenario["calc_results_mock"],
                    manual_inputs=cenario["inputs_manual"]
                )
                
                # 2. Validações (Assertions)
                
                # A. Validação de Metodologia e Texto Jurídico
                self.assertEqual(context["programa"]["metodologia"], cenario["expected"].get("modelo_txt"), 
                                 f"Falha no Nome do Modelo em {cenario['id']}")
                
                # B. Validação da Tabela de Fair Value (Preço Unitário Formatado)
                if "fv_unitario" in cenario["expected"]:
                    # Pega o primeiro item da tabela de resultados
                    fv_gerado = context["tabelas"]["resultados_fair_value"][0]["fv_final"]
                    self.assertEqual(fv_gerado, cenario["expected"]["fv_unitario"],
                                     "Falha na formatação ou valor do Fair Value Unitário")

                # C. Validação de Quantidade de Beneficiários (Nova Correção)
                if "qtd_beneficiarios" in cenario["expected"]:
                    # Verifica na tabela de cronograma se a quantidade está lá
                    qtd_gerada = context["tabelas"]["cronograma"][0]["qtd"]
                    self.assertEqual(str(qtd_gerada), str(cenario["expected"]["qtd_beneficiarios"]),
                                     "Falha na Quantidade de Beneficiários na tabela")

                # D. Validação de Campos Específicos (Privado vs Público)
                if "metodo_privado" in cenario["expected"]:
                    self.assertEqual(context["calculo"]["metodo_precificacao_privado"], cenario["expected"]["metodo_privado"],
                                     "Falha no método de precificação privado")
                
                # E. Validação da Moeda (DI vs Bond)
                if "moeda_ref" in cenario["expected"]:
                    self.assertEqual(context["calculo"]["moeda"], cenario["expected"]["moeda_ref"],
                                     "Falha na seleção da Moeda (DI vs T-Bond)")

                # F. Validação de Textos Condicionais (Performance)
                if "texto_perf" in cenario["expected"]:
                    self.assertIn(cenario["expected"]["texto_perf"], context["regras"]["texto_performance"],
                                  "Falha na descrição da regra de performance")

                print("OK ✅")

if __name__ == '__main__':
    unittest.main()
