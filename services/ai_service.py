"""
Serviço de Integração com IA e Processamento de Documentos.

Responsável por:
1. Extrair texto de arquivos (PDF/DOCX).
2. Comunicar com a API do Google Gemini.
3. Gerar estruturas de dados (PlanAnalysisResult) com foco em IFRS 2 / CPC 10.
4. Classificar instrumentos (Equity vs Cash) e extrair prazos de vencimento distintos de vesting.
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
    @st.cache_data(show_spinner=False)
    def analyze_plan_with_gemini(text: str, api_key: str) -> Optional[PlanAnalysisResult]:
        """
        Envia o texto do contrato para o Gemini e converte a resposta JSON
        em um objeto tipado PlanAnalysisResult, focado em classificação contábil.
        """
        if not HAS_GEMINI or not api_key:
            return None
            
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Prompt Otimizado para IFRS 2 / CPC 10 e Classificação Contábil
        prompt = f"""
        Você é um Consultor Sênior em Remuneração Executiva e Especialista em IFRS 2 (CPC 10).
        Sua tarefa é analisar o contrato fornecido e gerar um JSON estruturado para precificação e contabilização.

        DIRETRIZES DE ANÁLISE CRÍTICA (IFRS 2):
        
        1. **Classificação (Settlement Type):**
           - **EQUITY_SETTLED:** Se o plano entrega AÇÕES reais da empresa.
           - **CASH_SETTLED:** Se o plano paga em DINHEIRO baseado no valor da ação (Phantom Shares, SARs, Múltiplos). Procure termos como "Liquidação Financeira", "Pagamento em Caixa", "Direito de Valorização".
           - **HYBRID:** Se a empresa tem a opção de escolher como pagar.

        2. **Prazos (Vesting vs Expiration):**
           - **Vesting (Carência):** Período para ganhar o direito (ex: 3 anos, 25% ao ano).
           - **Expiration (Vencimento/Life):** Prazo máximo contratual para exercer a opção (ex: 10 anos). 
           - *Nota:* Frequentemente o vesting é curto (3-4 anos) mas o vencimento é longo (10 anos). Diferencie-os.

        3. **Instrumento e Modelo:**
           - Strike Zero/Simbólico -> RSU (Modelo RSU).
           - Strike de Mercado -> Opção (Black-Scholes ou Binomial).
           - Gatilho de Performance (TSR) -> Monte Carlo.

        TEXTO DO CONTRATO:
        {text[:90000]}

        SAÍDA JSON (ESTRITA):
        {{
            "program_summary": "Resumo Markdown focado em RH/Jurídico. Ex: '**Instrumento:** Phantom Shares (Caixa)...'",
            
            "valuation_params": "Resumo Markdown focado em Quant. Ex: '**1. Liquidação:** Caixa (Passivo)... **2. Life:** 10 anos...'",
            
            "summary": "Parágrafo curto geral.",
            
            "contract_features": "Lista curta das principais cláusulas.",
            
            "model_data": {{
                "recommended_model": "RSU" | "Binomial" | "Black-Scholes" | "Monte Carlo",
                "settlement_type": "EQUITY_SETTLED" | "CASH_SETTLED" | "HYBRID",
                "deep_rationale": "Justificativa técnica. Se for Cash-Settled, mencione a necessidade de remensuração a cada balanço.",
                "justification": "Frase curta para UI.",
                "comparison": "Comparação breve.",
                "pros": ["Pró 1"], 
                "cons": ["Contra 1"],
                "params": {{
                    "option_life": <float, Prazo TOTAL de vencimento do contrato em anos (ex: 10.0)>,
                    "strike_price": <float>,
                    "strike_is_zero": <bool, true se for RSU/Matching/Phantom com custo zero>,
                    "turnover_rate": <float, ex: 0.05 para 5%>,
                    "early_exercise_factor": <float, geralmente 2.0>,
                    "lockup_years": <float, anos de restrição pós-vesting>,
                    "has_strike_correction": <bool>,
                    "has_market_condition": <bool>,
                    "vesting_schedule": [
                        {{
                            "period_years": <float, Data do Vesting>, 
                            "percentage": <float, ex: 0.25>,
                            "expiration_years": <float, Data de Vencimento desta tranche (opcional, se diferente do geral)>
                        }}
                    ]
                }}
            }}
        }}
        """
        
        try:
            response = model.generate_content(prompt)
            clean_text = re.sub(r'```json|```', '', response.text).strip()
            data = json.loads(clean_text)
            
            return DocumentService._map_json_to_domain(data)
            
        except Exception as e:
            print(f"Erro na API Gemini: {e}")
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

        # Mapeamento do Enum de Liquidação (Settlement)
        settlement_str = model_data.get('settlement_type', 'EQUITY_SETTLED').upper()
        settlement_enum = SettlementType.EQUITY_SETTLED # Default seguro
        
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
                # Se expiration não vier na tranche, usa o global
                p_exp = float(t.get('expiration_years', global_life))
                if p_exp < p_y: p_exp = global_life # Correção de sanidade
                
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
            settlement_type=settlement_enum, # Novo campo obrigatório
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
            code = response.text
            code = re.sub(r'^```python\s*', '', code)
            code = re.sub(r'^```\s*', '', code)
            code = re.sub(r'\s*```$', '', code)
            return code.strip()
        except Exception as e:
            return f"# Erro na geração de código: {e}"

    @staticmethod
    def mock_analysis(text: str) -> PlanAnalysisResult:
        """Gera dados fictícios para demonstração (sem API Key)."""
        # Mock de um plano de Phantom Shares (Cash Settled)
        tranches = [
            Tranche(vesting_date=1.0, proportion=0.33, expiration_date=10.0),
            Tranche(vesting_date=2.0, proportion=0.33, expiration_date=10.0),
            Tranche(vesting_date=3.0, proportion=0.34, expiration_date=10.0)
        ]
        
        return PlanAnalysisResult(
            summary="[MOCK] Plano Phantom Shares: Liquidação em Caixa.",
            program_summary="**Instrumento:** Phantom Shares (Direito de Valorização).\n\n**Liquidação:** Financeira (Cash-Settled) - Passivo Contábil.",
            valuation_params="**1. Classificação:** Passivo (Remensurar a cada balanço).\n\n**2. Vencimento:** 10 anos (Life) vs Vesting 3 anos.",
            contract_features="[MOCK] Vesting 3 anos, Life 10 anos, Pagamento em Dinheiro.",
            methodology_rationale="[MOCK] Por ser liquidado em caixa (Phantom), deve ser tratado como passivo e remensurado a valor justo. Modelo Binomial recomendado se houver barreiras, ou BS Graded para casos simples.",
            model_recommended=PricingModelType.BLACK_SCHOLES_GRADED,
            settlement_type=SettlementType.CASH_SETTLED, # Mockando Cash Settled
            model_reason="[MOCK] Phantom Shares = Cash Settled.",
            model_comparison="[MOCK] Remensuração obrigatória.",
            pros=["Flexibilidade para o funcionário"], 
            cons=["Impacto no Caixa da empresa"],
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
