"""
Serviço de Integração com IA e Processamento de Documentos.

Responsável por:
1. Extrair texto de arquivos (PDF/DOCX).
2. Comunicar com a API do Google Gemini.
3. Gerar estruturas de dados (PlanAnalysisResult) a partir de texto não estruturado.
4. Gerar código Python dinâmico para simulações complexas.
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

from core.domain import PlanAnalysisResult, Tranche, PricingModelType

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
        em um objeto tipado PlanAnalysisResult.
        """
        if not HAS_GEMINI or not api_key:
            return None
            
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Prompt otimizado para extração JSON
        prompt = f"""
        - Você é um Atuário Sênior e Especialista em IFRS 2. Sua filosofia é a PARCIMÔNIA: Utilize sempre o modelo mais simples possível.
        - Monte Carlo: APENAS se houver 'Market Conditions' (ex: gatilhos de TSR, preço da ação alvo). NÃO recomende Monte Carlo apenas por causa de vesting acelerado ou turnover; isso se ajusta na quantidade/prazo, não no modelo de preço.
        - RSU/Matching: Se Strike == 0 (ou simbólico), o modelo É 'RSU' (Valor da ação descontado de dividendos). Tranches diferentes são tratadas separadamente.
        - Binomial: Para opções com lock-up ou exercício antecipado.
        
        TEXTO:
        {text[:45000]}

        Extraia os parâmetros para precificação de opções.
        SAÍDA JSON (ESTRITA):
        {{
            "program_summary": "Resumo executivo do programa: Quem recebe, qual o objetivo, quantidade total de instrumentos e regras de desligamento (Bad/Good leaver) de forma resumida.",
            "deep_rationale": "Justificativa técnica alinhada à filosofia de parcimônia. Se for RSU, explique que é devido à ausência de opcionalidade (Strike Zero).",
            "summary": "Resumo geral.",
            "model_data": {{
                "recommended_model": "RSU" | "Binomial" | "Black-Scholes" | "Monte Carlo",
                "deep_rationale": "Justificativa técnica alinhada à filosofia de parcimônia. Se for RSU, explique que é devido à ausência de opcionalidade (Strike Zero).",
                "justification": "Frase curta.",
                "comparison": "Comparação com outros modelos.",
                "pros": ["Pró 1"], 
                "cons": ["Contra 1"],
                "params": {{
                    "option_life": <float>,
                    "strike_price": <float>,
                    "strike_is_zero": <bool>,
                    "turnover_rate": <float>,
                    "early_exercise_factor": <float>,
                    "lockup_years": <float>,
                    "has_strike_correction": <bool>,
                    "has_market_condition": <bool>,
                    "vesting_schedule": [
                        {{"period_years": <float>, "percentage": <float>}}
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
            # Em produção, logar o erro. Aqui retornamos o Mock por segurança/fallback
            print(f"Erro na API Gemini: {e}")
            return DocumentService.mock_analysis(text)

    @staticmethod
    def _map_json_to_domain(data: Dict[str, Any]) -> PlanAnalysisResult:
        """Helper privado para converter dicionário JSON em objeto de Domínio."""
        model_data = data.get('model_data', {})
        params = model_data.get('params', {})
        
        # Mapeamento do Enum
        rec_model_str = model_data.get('recommended_model', '').upper().replace(" ", "_")
        model_enum = PricingModelType.UNDEFINED
        
        for m in PricingModelType:
            if m.name in rec_model_str:
                model_enum = m
                break
        
        # Fallback para Black Scholes Graded se a IA falar apenas "Black Scholes"
        if model_enum == PricingModelType.UNDEFINED and "BLACK" in rec_model_str:
            model_enum = PricingModelType.BLACK_SCHOLES_GRADED

        # Construção das Tranches
        tranches = []
        for t in params.get('vesting_schedule', []):
            try:
                p_y = float(t.get('period_years', 0))
                p_p = float(t.get('percentage', 0))
                if p_y > 0: tranches.append(Tranche(p_y, p_p))
            except (ValueError, TypeError):
                continue

        def safe_float(val, default=0.0):
            try: return float(val)
            except: return default

        return PlanAnalysisResult(
            summary=data.get('summary', ''),
            contract_features=data.get('contract_features', ''),
            methodology_rationale=model_data.get('deep_rationale', ''),
            model_recommended=model_enum,
            model_reason=model_data.get('justification', ''),
            model_comparison=model_data.get('comparison', ''),
            pros=model_data.get('pros', []),
            cons=model_data.get('cons', []),
            tranches=tranches,
            option_life_years=safe_float(params.get('option_life'), 5.0),
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
        
        # Serialização segura de parâmetros
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
        tranches = [Tranche(1.0, 0.25), Tranche(2.0, 0.25), Tranche(3.0, 0.25), Tranche(4.0, 0.25)]
        return PlanAnalysisResult(
            summary="[MOCK] Plano simulado: 4 tranches, correção de strike (IGPM).",
            contract_features="[MOCK] Vesting 4 anos (25% a.a), Correção Monetária Strike.",
            methodology_rationale="[MOCK] Binomial recomendado devido a barreiras complexas.",
            model_recommended=PricingModelType.BINOMIAL, 
            model_reason="[MOCK] Strike Dinâmico e Lock-up detectados.",
            model_comparison="[MOCK] Binomial captura melhor a dinâmica de lock-up.",
            pros=["Trata Lock-up", "Inflação"], 
            cons=["Custo Computacional"],
            tranches=tranches,
            has_strike_correction=True,
            option_life_years=10.0,
            strike_price=10.0,
            lockup_years=2.0,
            turnover_rate=0.05,
            early_exercise_multiple=2.5
        )
