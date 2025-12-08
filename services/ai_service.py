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
        
        # Prompt Atualizado com Lógica de IFRS 2 e Valuation Explícito
        prompt = f"""
        Você é um Consultor Sênior em Remuneração Executiva e Especialista em IFRS 2 (CPC 10).
        Sua tarefa é analisar o contrato fornecido e gerar um JSON estruturado.
        
        CONTEXTO DE ANÁLISE (DIRETRIZES RÍGIDAS):
        1. **Resumo do Programa**: Deve ser detalhado e categorizado (Instrumento, Vesting, Liquidação, Forfeiture, Aceleração, Lock-up).
        2. **Parâmetros de Valuation (CRÍTICO)**: Não apenas liste os números. Você DEVE EXPLICITAR O IMPACTO NO FAIR VALUE (FV) para o usuário final da ferramenta:
           - **Instrumento**: Se Ações/Opções (Equity-settled) -> FV fixado na Outorga (não remensura). Se Dinheiro (Cash-settled) -> Remensurado.
           - **Vesting de Tempo**: Citar que não ajusta o FV unitário, mas impacta a quantidade (Turnover).
           - **Graded Vesting (Tranches)**: Explicitar que "O cálculo do FV deve ser segregado por Tranche individual".
           - **Performance de Mercado (TSR)**: "Deve ser incorporada NO cálculo do FV (Monte Carlo)".
           - **Performance Não-Mercado (EBITDA)**: "Não impacta o FV unitário, ajusta-se a quantidade esperada (Vesting Condition)".
           - **Lock-up**: "Deve ser aplicado desconto de iliquidez (Chaffe) sobre o FV".
           - **Forfeiture/Turnover**: "Ajusta a quantidade de instrumentos, aplicado fora do modelo de precificação unitária".

        TEXTO DO CONTRATO:
        {text[:45000]}

        SAÍDA JSON (ESTRITA):
        {{
            "program_summary": "Resumo estruturado em tópicos: Instrumento, Condições de Vesting, Cronograma, Liquidação (Ações/Caixa), Regras de Forfeiture (Good/Bad Leaver), Aceleração (Change of Control) e Lock-up.",
            
            "valuation_params": "Texto explicativo focado no IFRS 2. Exemplo: '1. Instrumento: Opções (Equity-settled), FV fixado na data de outorga. 2. Vesting: Graded (3 tranches), exige cálculo de FV individual para cada tranche. 3. Lock-up: Identificado (2 anos), aplicar desconto de iliquidez (Chaffe). 4. Performance: Não há gatilhos de mercado, utilizar Black-Scholes ou Binomial padrão.'",
            
            "summary": "Um parágrafo curto resumindo o plano.",
            
            "contract_features": "Lista curta das principais cláusulas.",
            
            "model_data": {{
                "recommended_model": "RSU" | "Binomial" | "Black-Scholes" | "Monte Carlo",
                "deep_rationale": "Justificativa técnica da escolha do modelo baseada na PARCIMÔNIA. Se Strike for Zero (Matching/RSU), USE 'RSU'. Se houver TSR, USE 'Monte Carlo'. Se houver Lock-up/Americanas, USE 'Binomial'.",
                "justification": "Frase curta para UI.",
                "comparison": "Comparação breve.",
                "pros": ["Pró 1"], 
                "cons": ["Contra 1"],
                "params": {{
                    "option_life": <float, Estimativa de vida da opção em anos>,
                    "strike_price": <float>,
                    "strike_is_zero": <bool, true se for RSU/Matching Shares>,
                    "turnover_rate": <float, ex: 0.05 para 5%>,
                    "early_exercise_factor": <float, geralmente 2.0>,
                    "lockup_years": <float, anos de restrição pós-vesting>,
                    "has_strike_correction": <bool>,
                    "has_market_condition": <bool>,
                    "vesting_schedule": [
                        {{"period_years": <float>, "percentage": <float, ex: 0.25>}}
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
            program_summary=data.get('program_summary', 'Resumo do programa não identificado.'), # <--- NOVO
            valuation_params=data.get('valuation_params', 'Parâmetros não identificados.'),      # <--- NOVO
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
