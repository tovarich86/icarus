"""
Serviço de Integração com IA e Processamento de Documentos.
Versão Corrigida: Tratamento robusto de erros JSON e parser tolerante.
"""

import json
import re
from typing import Optional, Dict, Any
import streamlit as st

# Importações condicionais para bibliotecas pesadas
try:
    import PyPDF2
    from docx import Document
    HAS_DOC_LIBS = True
except ImportError:
    HAS_DOC_LIBS = False

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

from core.domain import PlanAnalysisResult, Tranche, PricingModelType, SettlementType

class DocumentService:
    """
    Fachada para serviços de leitura de documentos e geração via LLM.
    Inclui tratamento robusto para falhas de geração de JSON.
    """

    @staticmethod
    def extract_text(uploaded_file) -> str:
        """
        Extrai texto bruto de objetos de arquivo do Streamlit (BytesIO).
        Suporta PDF e DOCX.
        """
        if not HAS_DOC_LIBS:
            return "Erro: Bibliotecas PyPDF2 ou python-docx não instaladas."
            
        text = ""
        filename = uploaded_file.name.lower()
        
        try:
            if filename.endswith('.pdf'):
                reader = PyPDF2.PdfReader(uploaded_file)
                for page in reader.pages:
                    content = page.extract_text()
                    if content: text += content
            elif filename.endswith('.docx'):
                doc = Document(uploaded_file)
                text = "\n".join([p.text for p in doc.paragraphs])
        except Exception as e:
            return f"Erro na leitura do arquivo {filename}: {str(e)}"
            
        return text

    @staticmethod
    def _sanitize_json_output(raw_text: str) -> str:
        """
        Limpa a saída do LLM para garantir que apenas o JSON seja processado.
        Remove blocos de código markdown (```json ... ```).
        """
        # Remove marcadores de código Markdown
        text = re.sub(r'```json\s*', '', raw_text)
        text = re.sub(r'```\s*$', '', text)
        return text.strip()

    @staticmethod
    @st.cache_data(show_spinner=False)
    def analyze_plan_with_gemini(text: str, api_key: str) -> Optional[PlanAnalysisResult]:
        """
        Envia o texto do contrato para o Gemini e converte a resposta JSON
        em um objeto tipado PlanAnalysisResult.
        """
        if not HAS_GEMINI or not api_key:
            return None
            
        genai.configure(api_key=api_key)
        generation_config = {
            "temperature": 0.1, # Reduz criatividade desnecessária, foca em precisão
            "response_mime_type": "application/json" # Força saída JSON nativa
        }

        model = genai.GenerativeModel(
            model_name='gemini-2.5-flash-lite', # Use explicitamente o 1.5 Flash para velocidade máxima
            generation_config=generation_config
        )
        
        # Prompt Reforçado para evitar erros de sintaxe JSON
        # 1. A saída deve ser APENAS um JSON válido.
        # 2. NÃO use quebras de linha reais dentro das strings. Use o caractere de escape \\n literal.
        prompt = f"""
        Você é um Especialista em IFRS 2 (CPC 10). Analise o contrato e gere um JSON para precificação.
        REGRAS DE EXTRAÇÃO DE TEXTO ("valuation_params"):
        1. A PRIMEIRA LINHA de "valuation_params" DEVE SER OBRIGATORIAMENTE: "Modelo Recomendado: [Nome do Modelo]"
        2. Em seguida, liste: Strike, Spot (se houver), preço do ativo  subjacente, Volatilidade implícita no texto,carência e vida de cada tranch Vida da Opção, indicador de performance de mercado e/ou não mercado se houver, .
        3. Tratamento de Dividendos: Explique claramente se o yield deve ser considerado (q > 0) ou se há proteção de strike (q = 0).

        ATENÇÃO CRÍTICA À FORMATAÇÃO JSON:
        
        1. Se houver aspas duplas " dentro de um texto, elas DEVEM ser escapadas como \\".
        2. Verifique se todas as chaves e valores estão fechados corretamente.

        DIRETRIZES TÉCNICAS:
        - Classificação: EQUITY_SETTLED (Ações) vs CASH_SETTLED (Caixa/Phantom).
        - Dividendos: Se o participante não recebe dividendos na carência, o modelo deve descontar o yield.
        - Prazos: Diferencie Vesting (Carência) de Expiration (Vencimento total).

        TEXTO DO CONTRATO:
        {text[:80000]}

        SAÍDA JSON (ESTRITA):
        {{
            "program_summary": "Resumo Markdown usando \\n\\n para parágrafos.",
            "valuation_params": "Resumo técnico usando \\n\\n. Incluir análise de Dividendos na carência.",
            "summary": "Resumo geral curto.",
            "contract_features": "Principais cláusulas.",
            "model_data": {{
                "recommended_model": "RSU" | "Binomial" | "Black-Scholes" | "Monte Carlo",
                "settlement_type": "EQUITY_SETTLED" | "CASH_SETTLED" | "HYBRID",
                "deep_rationale": "Justificativa técnica.",
                "justification": "Frase curta.",
                "comparison": "Comparação breve.",
                "pros": ["Pró 1"], 
                "cons": ["Contra 1"],
                "params": {{
                    "option_life": <float>,
                    "strike_price": <float>,
                    "strike_is_zero": <bool>,
                    "dividends_during_vesting": <bool>,
                    "turnover_rate": <float>,
                    "early_exercise_factor": <float>,
                    "lockup_years": <float>,
                    "has_strike_correction": <bool>,
                    "has_market_condition": <bool>,
                    "vesting_schedule": [
                        {{
                            "period_years": <float>, 
                            "percentage": <float>,
                            "expiration_years": <float>
                        }}
                    ]
                }}
            }}
        }}
        """
        
        try:
            response = model.generate_content(prompt)
            
            # 1. Sanitização
            clean_text = DocumentService._sanitize_json_output(response.text)
            
            # 2. Parse Tolerante (strict=False permite quebras de linha dentro de strings)
            data = json.loads(clean_text, strict=False)
            
            return DocumentService._map_json_to_domain(data)
            
        except json.JSONDecodeError as e:
            # Captura erros de parsing específicos e mostra o que a IA gerou para debug
            st.error(f"⚠️ Erro de Sintaxe no JSON gerado pela IA: {e}")
            with st.expander("Ver JSON Bruto (Debug)", expanded=False):
                st.code(clean_text, language='json')
            st.warning("Usando dados de exemplo (Mock) para não interromper o fluxo.")
            return DocumentService.mock_analysis(text)
            
        except Exception as e:
            st.error(f"⚠️ Erro Geral na Análise IA: {str(e)}")
            return DocumentService.mock_analysis(text)

    @staticmethod
    def _map_json_to_domain(data: Dict[str, Any]) -> PlanAnalysisResult:
        """Helper privado para converter dicionário JSON em objeto de Domínio."""
        model_data = data.get('model_data', {})
        params = model_data.get('params', {})
        
        # Mapeamento do Enum de Modelo
        rec_model_str = model_data.get('recommended_model', '').upper().replace(" ", "_")
        model_enum = PricingModelType.UNDEFINED
        
        for m in PricingModelType:
            if m.name in rec_model_str:
                model_enum = m
                break
        
        if model_enum == PricingModelType.UNDEFINED and "BLACK" in rec_model_str:
            model_enum = PricingModelType.BLACK_SCHOLES_GRADED

        # Mapeamento do Enum de Liquidação
        settlement_str = model_data.get('settlement_type', 'EQUITY_SETTLED').upper()
        settlement_enum = SettlementType.EQUITY_SETTLED 
        
        if "CASH" in settlement_str:
            settlement_enum = SettlementType.CASH_SETTLED
        elif "HYBRID" in settlement_str:
            settlement_enum = SettlementType.HYBRID

        # Construção das Tranches
        tranches = []
        global_life = safe_float(params.get('option_life'), 5.0)
        
        for t in params.get('vesting_schedule', []):
            try:
                p_y = float(t.get('period_years', 0))
                p_p = float(t.get('percentage', 0))
                p_exp = float(t.get('expiration_years', global_life))
                if p_exp < p_y: p_exp = global_life 
                
                if p_y > 0: 
                    tranches.append(Tranche(vesting_date=p_y, proportion=p_p, expiration_date=p_exp))
            except (ValueError, TypeError):
                continue

        return PlanAnalysisResult(
            summary=data.get('summary', ''),
            program_summary=data.get('program_summary', 'Resumo do programa não identificado.'),
            valuation_params=data.get('valuation_params', 'Parâmetros não identificados.'),
            contract_features=data.get('contract_features', ''),
            methodology_rationale=model_data.get('deep_rationale', ''),
            model_recommended=model_enum,
            settlement_type=settlement_enum,
            model_reason=model_data.get('justification', ''),
            model_comparison=model_data.get('comparison', ''),
            pros=model_data.get('pros', []),
            cons=model_data.get('cons', []),
            tranches=tranches,
            option_life_years=global_life,
            strike_price=safe_float(params.get('strike_price'), 0.0),
            strike_is_zero=bool(params.get('strike_is_zero', False)),
            turnover_rate=safe_float(params.get('turnover_rate'), 0.0),
            early_exercise_multiple=safe_float(params.get('early_exercise_factor'), 2.0),
            lockup_years=safe_float(params.get('lockup_years'), 0.0),
            has_strike_correction=bool(params.get('has_strike_correction', False)),
            has_market_condition=bool(params.get('has_market_condition', False))
        )

    @staticmethod
    def generate_custom_monte_carlo_code(contract_text: str, params: Dict, api_key: str) -> str:
        """Gera código Python customizado via IA para simulação de Monte Carlo."""
        if not api_key: 
            return "# Erro: API Key necessária."
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        safe_params = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in params.items()}
        
        prompt = f"""
        Escreva um script Python EXECUÁVEL para precificar opções usando Monte Carlo.
        
        REQUISITOS:
        1. Use 'numpy'.
        2. Defina uma função `run_simulation()`.
        3. No final, chame `fv = run_simulation()` e `print(fv)`.
        
        PARÂMETROS: {json.dumps(safe_params, default=str)}
        CONTEXTO: {contract_text[:10000]}
        
        SAÍDA: Apenas código Python.
        """
        
        try:
            response = model.generate_content(prompt)
            code = DocumentService._sanitize_json_output(response.text) # Reutiliza a limpeza
            return code
        except Exception as e:
            return f"# Erro na geração de código: {e}"

    @staticmethod
    def mock_analysis(text: str) -> PlanAnalysisResult:
        """Gera dados fictícios para demonstração (sem API Key ou em caso de erro)."""
        tranches = [
            Tranche(vesting_date=1.0, proportion=0.33, expiration_date=10.0),
            Tranche(vesting_date=2.0, proportion=0.33, expiration_date=10.0),
            Tranche(vesting_date=3.0, proportion=0.34, expiration_date=10.0)
        ]
        
        return PlanAnalysisResult(
            summary="[MOCK - ERRO NA IA] Plano Phantom Shares Simulados.",
            program_summary="**Atenção:** Os dados abaixo são fictícios pois houve um erro na leitura do contrato pela IA.\n\n**Instrumento:** Phantom Shares.\n\n**Liquidação:** Financeira.",
            valuation_params="**1. Status:** Dados de Fallback.\n\n**2. Dividendos:** Não recebe.\n\n**3. Life:** 10 anos.",
            contract_features="[MOCK] Vesting 3 anos, Life 10 anos, Pagamento em Dinheiro.",
            methodology_rationale="[MOCK] Ocorreu um erro técnico na geração do racional pela IA. O sistema carregou este modelo padrão para permitir que você continue o teste da ferramenta manualmente.",
            model_recommended=PricingModelType.BLACK_SCHOLES_GRADED,
            settlement_type=SettlementType.CASH_SETTLED,
            model_reason="[MOCK] Dados de contingência.",
            model_comparison="[MOCK] Sem comparação disponível.",
            pros=["Continuidade do teste"], 
            cons=["Dados não reais"],
            tranches=tranches,
            has_strike_correction=False,
            option_life_years=10.0,
            strike_price=15.0,
            lockup_years=0.0,
            turnover_rate=0.05,
            early_exercise_multiple=2.0
        )

def safe_float(val, default=0.0):
    try: return float(val)
    except: return default
