"""
Serviço de Integração com IA e Processamento de Documentos.
Versão Refatorada: Integração Nativa com Pydantic e Schemas JSON.
"""

import json
import re
import logging
from typing import Optional, Dict, Any
import streamlit as st

# Importações condicionais
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

# Domínio e Regras
from core.domain import PlanAnalysisResult, Tranche, PricingModelType, SettlementType
from services.rule_extractor import RuleBasedExtractor

class DocumentService:
    """
    Fachada para leitura de documentos e inteligência (Híbrida: Regras + LLM).
    """
    
    # Instância única do extrator (Performance)
    _rule_extractor = RuleBasedExtractor()

    @staticmethod
    def extract_text(uploaded_file) -> str:
        """Extrai texto de PDF/DOCX de forma resiliente."""
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
    def analyze_plan_hybrid(text: str, api_key: str = None, use_ai: bool = True) -> PlanAnalysisResult:
        """
        Orquestrador:
        1. Executa Regras (Regex) primeiro.
        2. Se tiver API Key e use_ai=True, chama o LLM usando o contexto das regras.
        """
        # 1. Extração Determinística (Regras)
        rule_data = DocumentService._rule_extractor.analyze_single_plan(text)
        
        # 2. Se não for usar IA, converte o resultado das regras para o domínio e retorna
        if not use_ai or not api_key:
            return DocumentService._convert_rules_to_domain(rule_data)
        
        # 3. Chama IA com contexto
        return DocumentService.analyze_plan_with_gemini(text, api_key, rule_context=rule_data)

    @staticmethod
    def _convert_rules_to_domain(rule_data: dict) -> PlanAnalysisResult:
        """
        Converte a saída do RuleExtractor para o Objeto Pydantic oficial.
        """
        facts = rule_data.get("extracted_facts", {})
        plan_types = rule_data.get("detected_plan_types", [])
        
        main_type = plan_types[0] if plan_types else "Não Classificado"
        has_market = facts.get("has_tsr") or facts.get("has_market_condition")
        
        # Criação segura com Pydantic
        # Nota: Pydantic exige campos obrigatórios, então preenchemos com defaults sensatos
        return PlanAnalysisResult(
            summary=f"Análise via Regras: Detectado {main_type}.",
            program_summary=f"Plano identificado por palavras-chave como {main_type}.",
            valuation_params=f"* **Vesting:** {facts.get('vesting_period', 3.0)} anos\n* **Lock-up:** {facts.get('lockup_years', 0.0)} anos",
            contract_features="; ".join(plan_types),
            methodology_rationale="Metodologia inferida por regras (Regex).",
            
            model_recommended=PricingModelType.MONTE_CARLO if has_market else PricingModelType.BLACK_SCHOLES_GRADED,
            settlement_type=SettlementType.EQUITY_SETTLED, # Default seguro
            
            tranches=[
                Tranche(
                    vesting_date=float(facts.get('vesting_period', 3.0)), 
                    proportion=1.0,
                    expiration_date=10.0
                )
            ],
            
            has_market_condition=bool(has_market),
            lockup_years=float(facts.get('lockup_years', 0.0)),
            has_strike_correction=False
        )

    @staticmethod
    def _sanitize_json_output(raw_text: str) -> str:
        """Limpeza básica de Markdown code blocks."""
        text = re.sub(r'```json\s*', '', raw_text)
        text = re.sub(r'```\s*$', '', text)
        return text.strip()

    @staticmethod
    @st.cache_data(show_spinner=False)
    def analyze_plan_with_gemini(text: str, api_key: str, rule_context: dict = None) -> Optional[PlanAnalysisResult]:
        """
        Analisa o plano usando Gemini com SAÍDA ESTRUTURADA (JSON Schema).
        """
        if not HAS_GEMINI or not api_key:
            return None
            
        genai.configure(api_key=api_key)
        
        # Configuração para JSON mode
        generation_config = {
            "temperature": 0.2, # Baixa temperatura para precisão
            "response_mime_type": "application/json"
        }

        model = genai.GenerativeModel(
            model_name='gemini-2.5-flash', # ou flash-lite
            generation_config=generation_config
        )
        
        # --- A MÁGICA: Extração Automática do Schema do Pydantic ---
        # A IA recebe a definição exata da classe que criamos no domain.py
        target_schema = json.dumps(PlanAnalysisResult.model_json_schema(), indent=2)
        
        context_str = ""
        if rule_context:
            facts = rule_context.get("extracted_facts", {})
            types = rule_context.get("detected_plan_types", [])
            context_str = f"""
            CONTEXTO DE AUDITORIA (Regras já processadas):
            - Tipos detectados: {types}
            - Vesting (Regex): {facts.get('vesting_period')} anos
            - Lockup (Regex): {facts.get('lockup_years')} anos
            Use estes números se o texto for ambíguo.
            """

        prompt = f"""
        Você é um Atuário Sênior e Engenheiro de Software.
        Analise o contrato de Incentivo de Longo Prazo (ILP/Stock Options) abaixo.
        
        SUA TAREFA:
        Extrair os dados e preencher o seguinte JSON Schema.
        
        SCHEMA OBRIGATÓRIO (Siga rigorosamente os tipos e descrições):
        {target_schema}
        
        {context_str}
        
        CONTRATO (Trecho):
        {text[:90000]}
        """
        
        try:
            response = model.generate_content(prompt)
            clean_json = DocumentService._sanitize_json_output(response.text)
            
            # --- VALIDAÇÃO AUTOMÁTICA PYDANTIC ---
            # Se a IA alucinar um campo ou errar o tipo, isso explode aqui e evita erros silenciosos
            result = PlanAnalysisResult.model_validate_json(clean_json)
            
            return result
            
        except Exception as e:
            st.error(f"Erro na análise de IA: {str(e)}")
            # Fallback para regras se a IA falhar
            if rule_context:
                return DocumentService._convert_rules_to_domain(rule_context)
            return DocumentService.mock_analysis(text)

    @staticmethod
    def generate_custom_monte_carlo_code(contract_text: str, params: Dict, api_key: str) -> str:
        """Gera código Python para simulação (Mantido similar, apenas ajustes menores)."""
        if not api_key: return "# Erro: API Key necessária."
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Limpa params para garantir tipos simples
        safe_params = {k: (float(v) if isinstance(v, (int, float)) else str(v)) for k, v in params.items()}
        
        prompt = f"""
        Aja como um Quant Developer. Escreva um script Python para precificar opções via Monte Carlo.
        
        Requisitos:
        1. Use numpy vetorizado.
        2. Implemente a função `run_simulation() -> float`.
        3. No final, chame a função e salve em uma variável `fv`.
        
        Parâmetros Base:
        {json.dumps(safe_params, indent=2)}
        
        Contexto do Plano:
        {contract_text[:15000]}
        
        Saída: Apenas código Python válido.
        """
        
        try:
            response = model.generate_content(prompt)
            return DocumentService._sanitize_json_output(response.text)
        except Exception as e:
            return f"# Erro ao gerar código: {e}"

    @staticmethod
    def mock_analysis(text: str) -> PlanAnalysisResult:
        """Retorna objeto Mock válido em caso de falha total."""
        return PlanAnalysisResult(
            summary="[MOCK] Falha na análise. Dados simulados.",
            program_summary="Dados gerados para teste de interface devido a erro na API.",
            valuation_params="* **Status:** Mock Data",
            contract_features="Simulado",
            methodology_rationale="Fallback.",
            model_recommended=PricingModelType.BLACK_SCHOLES_GRADED,
            settlement_type=SettlementType.EQUITY_SETTLED,
            tranches=[Tranche(vesting_date=1.0, proportion=1.0, expiration_date=5.0)],
            option_life_years=5.0
        )
