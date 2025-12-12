"""
Serviço de Integração com IA e Processamento de Documentos.
Versão Híbrida: Integração de Motor de Regras (Regex) com IA Generativa.
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

# Importações do Domínio e Serviços
from core.domain import PlanAnalysisResult, Tranche, PricingModelType, SettlementType
from services.rule_extractor import RuleBasedExtractor  # Novo Motor de Regras

class DocumentService:
    """
    Fachada para serviços de leitura de documentos, extração baseada em regras 
    e geração via LLM.
    """

    # Instância estática do extrator para performance (evita recompilar regex)
    _rule_extractor = RuleBasedExtractor()

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
    def analyze_plan_hybrid(text: str, api_key: str = None, use_ai: bool = True) -> PlanAnalysisResult:
        """
        Função principal de orquestração.
        1. Executa extração Hard-Coded (Regras/Regex).
        2. Se use_ai=False, retorna o resultado das regras imediatamente.
        3. Se use_ai=True, usa os dados das regras como contexto para a IA refinar.
        """
        # 1. Extração Hard-Coded (Custo Zero e Alta Precisão para números)
        rule_data = DocumentService._rule_extractor.analyze_single_plan(text)
        
        # 2. Modo Apenas Regras (Sem chave de API ou opção desativada)
        if not use_ai or not api_key:
            return DocumentService._convert_rules_to_domain(rule_data)
        
        # 3. Modo Híbrido (Regras + IA)
        return DocumentService.analyze_plan_with_gemini(text, api_key, rule_context=rule_data)

    @staticmethod
    def _convert_rules_to_domain(rule_data: dict) -> PlanAnalysisResult:
        """
        Converte a saída do RuleBasedExtractor diretamente para o objeto de domínio.
        Usado quando a IA está desligada ou falha.
        """
        facts = rule_data.get("extracted_facts", {})
        plan_types = rule_data.get("detected_plan_types", [])
        topic_matches = rule_data.get("topic_matches", {})
        
        # Lógica Determinística para Defaults
        main_type = plan_types[0] if plan_types else "Plano Não Classificado"
        
        # Inferência de Liquidação
        settlement = SettlementType.EQUITY_SETTLED
        if any(term in main_type for term in ["Phantom", "SAR", "Bonus", "Financeira"]):
            settlement = SettlementType.CASH_SETTLED
        
        # Inferência de Modelo Baseada em Fatos
        model = PricingModelType.BLACK_SCHOLES_GRADED
        if facts.get("has_tsr") or facts.get("has_market_condition"):
            model = PricingModelType.MONTE_CARLO
        elif any(term in main_type for term in ["Restricted", "RSU", "Ações Restritas"]):
            model = PricingModelType.RSU
        
        # Criação de Tranches Simplificada
        vesting_avg = facts.get("vesting_period", 3.0)
        tranches = [
            Tranche(vesting_date=vesting_avg, proportion=1.0, expiration_date=10.0)
        ]
        
        return PlanAnalysisResult(
            summary=f"Plano identificado automaticamente via Regras: {main_type}.",
            program_summary=f"**Tipo Detectado:** {main_type}\n\n**Mecanismo:** Análise baseada em palavras-chave e expressões regulares (Regex).",
            valuation_params=f"* **Vesting (Regra):** {vesting_avg} anos.\n* **Lock-up (Regra):** {facts.get('lockup_years', 0.0)} anos.",
            contract_features=f"Tópicos Encontrados: {', '.join(topic_matches.values())}",
            methodology_rationale="Metodologia definida por regras estáticas. Recomenda-se validação manual ou uso de IA para maior detalhamento.",
            model_recommended=model,
            settlement_type=settlement,
            model_reason="Inferido a partir da classificação do tipo de plano e presença de gatilhos de performance.",
            model_comparison="N/A (Modo Regras)",
            pros=["Processamento Instantâneo", "Custo Zero", "Dados Numéricos Precisos"],
            cons=["Falta de Nuance Narrativa", "Pode perder cláusulas atípicas"],
            tranches=tranches,
            has_market_condition=facts.get("has_tsr", False),
            has_strike_correction=facts.get("has_strike_correction", False),
            option_life_years=10.0,
            strike_is_zero=(model == PricingModelType.RSU),
            lockup_years=facts.get('lockup_years', 0.0)
        )

    @staticmethod
    def _sanitize_json_output(raw_text: str) -> str:
        """Limpa a saída do LLM para garantir JSON válido."""
        text = re.sub(r'```json\s*', '', raw_text)
        text = re.sub(r'```\s*$', '', text)
        return text.strip()

    @staticmethod
    @st.cache_data(show_spinner=False)
    def analyze_plan_with_gemini(text: str, api_key: str, rule_context: dict = None) -> Optional[PlanAnalysisResult]:
        """
        Envia o texto para o Gemini, injetando o contexto das regras para aumentar a precisão.
        """
        if not HAS_GEMINI or not api_key:
            return None
            
        genai.configure(api_key=api_key)
        
        generation_config = {
            "temperature": 0.3, # Reduzida para ser mais factual
            "response_mime_type": "application/json",
            "max_output_tokens": 4000
        }

        model = genai.GenerativeModel(
            model_name='gemini-2.5-flash-lite', 
            generation_config=generation_config
        )
        
        # Construção do Contexto das Regras (Grounding)
        context_str = ""
        if rule_context:
            facts = rule_context.get("extracted_facts", {})
            types = rule_context.get("detected_plan_types", [])
            
            context_str = f"""
            ### DADOS DE AUDITORIA PRELIMINAR (IMPORTANTE):
            O sistema já analisou este documento via algoritmos matemáticos (Regex) e encontrou com ALTA CONFIANÇA:
            - Tipo de Plano Provável: {', '.join(types) if types else 'Não classificado'}
            - Vesting (Carência): {facts.get('vesting_period', 'Não identificado')} anos.
            - Lock-up: {facts.get('lockup_years', '0')} anos.
            - Possui Cláusula de Malus/Clawback? {'SIM' if facts.get('has_malus_clawback') else 'Não detectado'}.
            - Possui Condição de Mercado (TSR)? {'SIM' if facts.get('has_tsr') else 'Não'}.
            
            DIRETRIZ: Use estes dados como base. Se o texto for ambíguo, confie nestes números extraídos.
            """
        
        prompt = f"""
        Você é um Auditor Especialista em IFRS 2 (CPC 10). Analise o contrato e gere um JSON para precificação.
        
        {context_str}

        ### 1. ÁRVORE DE DECISÃO DE MODELO (CRÍTICO - CPC 10)
        A) CONDIÇÃO DE MERCADO (Market Condition)? [Ref: CPC 10, Apêndice A]
           - Gatilhos: Preço da ação, TSR, Comparação com Índices.
           - MODELO: "Monte Carlo".

        B) CONDIÇÃO DE NÃO-MERCADO? [Ref: CPC 10, Item 19]
           - Gatilhos: EBITDA, Lucro Líquido.
           - MODELO: "Black-Scholes" ou "Binomial".
           - REGRA: PROIBIDO "Monte Carlo".

        C) EXERCÍCIO ANTECIPADO? [Ref: CPC 10, B5]
           - Janela > 6 meses entre Vesting e Vencimento.
           - MODELO: "Binomial".

        D) STRIKE ZERO?
           - MODELO: "RSU".

        ### 2. ESTRUTURA DO JSON
        Gere um JSON estrito com os campos: "program_summary", "valuation_params", "summary", "contract_features", "model_data".
        
        No campo "model_data", inclua "recommended_model", "settlement_type" (EQUITY_SETTLED/CASH_SETTLED) e "params".
        Dentro de "params", extraia: "option_life", "strike_price", "vesting_schedule" (lista de objetos), etc.

        ### 3. CONTEXTO DO CONTRATO
        {text[:80000]}
        """
        
        try:
            response = model.generate_content(prompt)
            clean_text = DocumentService._sanitize_json_output(response.text)
            data = json.loads(clean_text, strict=False)
            return DocumentService._map_json_to_domain(data)
            
        except json.JSONDecodeError:
            st.warning("IA gerou JSON inválido. Usando fallback de Regras.")
            # Fallback inteligente: se a IA falhar no JSON, retorna o resultado das regras
            if rule_context:
                return DocumentService._convert_rules_to_domain(rule_context)
            return DocumentService.mock_analysis(text)
            
        except Exception as e:
            st.error(f"Erro na IA: {str(e)}")
            if rule_context:
                return DocumentService._convert_rules_to_domain(rule_context)
            return DocumentService.mock_analysis(text)

    @staticmethod
    def _map_json_to_domain(data: Dict[str, Any]) -> PlanAnalysisResult:
        """Helper privado para converter dicionário JSON da IA em objeto de Domínio."""
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
        if not api_key: return "# Erro: API Key necessária."
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        safe_params = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in params.items()}
        
        prompt = f"""
        Escreva um script Python EXECUÁVEL para precificar opções usando Monte Carlo.
        PARÂMETROS: {json.dumps(safe_params, default=str)}
        CONTEXTO: {contract_text[:10000]}
        SAÍDA: Apenas código Python, importando numpy.
        """
        
        try:
            response = model.generate_content(prompt)
            return DocumentService._sanitize_json_output(response.text)
        except Exception as e:
            return f"# Erro na geração de código: {e}"

    @staticmethod
    def mock_analysis(text: str) -> PlanAnalysisResult:
        """Gera dados fictícios (Mock) para demonstração em caso de erro fatal."""
        tranches = [
            Tranche(vesting_date=1.0, proportion=0.33, expiration_date=10.0),
            Tranche(vesting_date=2.0, proportion=0.33, expiration_date=10.0),
            Tranche(vesting_date=3.0, proportion=0.34, expiration_date=10.0)
        ]
        return PlanAnalysisResult(
            summary="[MOCK] Erro na análise. Dados fictícios carregados.",
            program_summary="**Erro:** Não foi possível processar o documento.\n\nDados simulados para teste de interface.",
            valuation_params="**Status:** Mock Data.",
            contract_features="[MOCK] Dados simulados.",
            methodology_rationale="[MOCK] Racional de contingência.",
            model_recommended=PricingModelType.BLACK_SCHOLES_GRADED,
            settlement_type=SettlementType.EQUITY_SETTLED,
            model_reason="Fallback do sistema.",
            model_comparison="N/A",
            pros=[], cons=[],
            tranches=tranches,
            option_life_years=10.0,
            strike_price=10.0
        )

def safe_float(val, default=0.0):
    try: return float(val)
    except: return default
