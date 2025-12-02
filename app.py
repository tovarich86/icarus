import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import io
import re
import json
from datetime import date
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
from enum import Enum

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="Icarus: Validador IFRS 2", layout="wide")

# =============================================================================
# CAMADA 1: DOMÍNIO E ESTRUTURAS DE DADOS
# =============================================================================

class PricingModelType(Enum):
    MONTE_CARLO = "Monte Carlo (Simulação)"
    BINOMIAL = "Binomial (Lattice Customizado)"
    BLACK_SCHOLES_GRADED = "Black-Scholes (Graded Vesting)"
    # BLACK_SCHOLES Vanilla removido conforme solicitado
    RSU = "RSU / Ações Restritas (Matching Shares)"
    UNDEFINED = "Indefinido"

@dataclass
class Tranche:
    """Representa uma tranche individual de um plano de vesting escalonado."""
    vesting_date: float # Anos até o vesting
    proportion: float   # % do total do grant (ex: 0.25)
    
@dataclass
class PlanAnalysisResult:
    # Metadados Qualitativos
    summary: str
    contract_features: str 
    methodology_rationale: str 
    model_recommended: PricingModelType
    model_reason: str
    model_comparison: str 
    pros: List[str]
    cons: List[str]
    
    # Parâmetros Quantitativos Extraídos
    tranches: List[Tranche] = field(default_factory=list)
    has_market_condition: bool = False 
    has_strike_correction: bool = False 
    option_life_years: float = 5.0 
    strike_price: float = 0.0
    strike_is_zero: bool = False
    turnover_rate: float = 0.0 # Taxa de saída (w)
    early_exercise_multiple: float = 2.0 # Fator M
    lockup_years: float = 0.0
    
    def get_avg_vesting(self) -> float:
        if not self.tranches: return 3.0
        return sum(t.vesting_date * t.proportion for t in self.tranches)

# =============================================================================
# CAMADA 2: MATEMÁTICA FINANCEIRA (ENGINES)
# =============================================================================

class FinancialMath:
    
    @staticmethod
    def bs_call(S, K, T, r, sigma, q=0.0):
        """
        Fórmula Black-Scholes-Merton robusta com tratamento de erros.
        Lida com casos de fronteira (Volatilidade Zero, Tempo Zero, Strike Zero).
        """
        # 1. Tratamento de Tempo (Vencido ou muito curto)
        if T <= 1e-6:
            return max(S - K, 0.0)
        
        # 2. Tratamento de Volatilidade Zero ou Negativa (Valor Intrínseco Descontado)
        if sigma <= 1e-6:
            val = (S * np.exp(-q * T)) - (K * np.exp(-r * T))
            return max(val, 0.0)
            
        # 3. Tratamento de Strike Zero ou Negativo (RSU / Forward)
        if K <= 1e-6:
            return S * np.exp(-q * T)
            
        # 4. Tratamento de Preço Spot Zero
        if S <= 1e-6:
            return 0.0

        try:
            d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        except Exception:
            # Fallback seguro em caso de overflow numérico
            return max(S - K, 0.0)

    @staticmethod
    def calculate_lockup_discount(volatility, lockup_time, stock_price, q):
        """Modelo Chaffe para desconto por falta de liquidez (Lock-up)."""
        if lockup_time <= 0: return 0.0
        
        vol_sq_t = (volatility ** 2) * lockup_time
        term_inner = 2 * (np.exp(vol_sq_t) - vol_sq_t - 1)
        
        if term_inner <= 0: return 0.0
        
        b = vol_sq_t + np.log(term_inner) - 2 * np.log(np.exp(vol_sq_t) - 1)
        if b < 0: return 0.0
        
        a = np.sqrt(b)
        discount_val = stock_price * np.exp(-q * lockup_time) * (norm.cdf(a / 2) - norm.cdf(-a / 2))
        
        return discount_val

    @staticmethod
    def binomial_custom_optimized(S, K, r, vol, q, vesting_years, turnover_w, multiple_M, hurdle_H, T_years, inflacao_anual, lockup_years, tipo_exercicio=0):
        """
        Versão vetorizada e otimizada do Modelo Binomial para alta performance.
        Remove loops aninhados para evitar travamento em T_years altos.
        """
        # Discretização Otimizada
        total_steps = int(T_years * 252)
        if total_steps > 2000: total_steps = 2000
        if total_steps < 10: total_steps = 10
        
        dt = T_years / total_steps
        
        # Índices de Vesting
        Nv = int(vesting_years / dt)
        if Nv > total_steps: Nv = total_steps
        
        # Parâmetros CRR
        u = np.exp(vol * np.sqrt(dt))
        d = 1.0 / u
        p = (np.exp((r - q) * dt) - d) / (u - d)
        
        # dtVesting ajustado (Conforme lógica VBA original)
        if total_steps > Nv:
            dt_vesting = 1.0 / (total_steps - Nv)
        else:
            dt_vesting = 0.0
            
        # 1. Árvore de Preços no Vencimento (Passo N)
        j_idx = np.arange(total_steps + 1)
        ST = S * (u ** (total_steps - j_idx)) * (d ** j_idx)
        
        # 2. Strike no Vencimento (Ajustado por Inflação)
        K_final = K * ((1 + inflacao_anual)**T_years)
        
        # 3. Valor da Opção no Vencimento
        vals_base = ST.copy()
        if lockup_years > 0:
            discounts = FinancialMath.calculate_lockup_discount(vol, lockup_years, ST, q)
            vals_base -= discounts
            
        payoffs = np.maximum(vals_base - K_final, 0.0)
        option_values = np.where(ST >= hurdle_H, payoffs, 0.0)
        
        # 4. Indução Retroativa (Backward Loop)
        for i in range(total_steps - 1, -1, -1):
            j_idx = np.arange(i + 1)
            S_node = S * (u ** (i - j_idx)) * (d ** j_idx)
            time_elapsed = i * dt
            K_curr = K * ((1 + inflacao_anual)**time_elapsed)
            
            hold_values = np.exp(-r * dt) * (p * option_values[:-1] + (1 - p) * option_values[1:])
            
            if i >= Nv:
                S_exerc = S_node.copy()
                if lockup_years > 0:
                    disc = FinancialMath.calculate_lockup_discount(vol, lockup_years, S_node, q)
                    S_exerc -= disc
                
                exercise_values = np.maximum(S_exerc - K_curr, 0.0)
                force_exercise_mask = (tipo_exercicio == 0) & (S_node > (multiple_M * K_curr))
                
                prob_ficar = np.exp(-turnover_w * dt_vesting)
                prob_sair = 1.0 - prob_ficar
                
                val_if_not_forced = (prob_sair * exercise_values) + (prob_ficar * hold_values)
                node_values = np.where(force_exercise_mask, exercise_values, val_if_not_forced)
                
                hurdle_mask = (S_node >= hurdle_H)
                val_hurdle_fail = prob_ficar * hold_values
                option_values = np.where(hurdle_mask, node_values, val_hurdle_fail)
            else:
                option_values = hold_values
                
        return option_values[0]

# =============================================================================
# CAMADA 3: SERVIÇOS (INTELIGÊNCIA E DECISÃO)
# =============================================================================

class ModelSelectorService:
    @staticmethod
    def select_model(analysis: PlanAnalysisResult) -> PlanAnalysisResult:
        if analysis.has_market_condition:
            analysis.model_recommended = PricingModelType.MONTE_CARLO
            # Ajuste de fallback se a IA não tiver preenchido
            if not analysis.methodology_rationale:
                analysis.methodology_rationale = "A presença de condições de mercado (ex: TSR) introduz dependência da trajetória do preço (Path Dependency), inviabilizando soluções fechadas (Black-Scholes) ou árvores simples (Binomial). O Monte Carlo é o único capaz de simular múltiplos cenários estocásticos para satisfazer essa condição."
            return analysis

        # --- LÓGICA AJUSTADA: Prioridade para RSU/Matching Shares se Strike for Zero ---
        if analysis.strike_is_zero:
            analysis.model_recommended = PricingModelType.RSU
            if not analysis.methodology_rationale:
                analysis.methodology_rationale = "O plano concede ações gratuitas (Matching Shares) ou com preço de exercício simbólico. Neste cenário, a 'opcionalidade' (direito de não exercer) é irrelevante, pois o ganho é sempre positivo. O modelo mais correto e direto é tratar como RSU, valorando pela cotação à vista descontada dos dividendos esperados durante o vesting. Caso existam restrições de venda pós-vesting (Lock-up), aplica-se um desconto de iliquidez (ex: Chaffe) sobre este valor base."
            return analysis

        gap_exercicio = analysis.option_life_years - analysis.get_avg_vesting()
        if analysis.has_strike_correction or gap_exercicio > 2.0 or analysis.lockup_years > 0:
            analysis.model_recommended = PricingModelType.BINOMIAL
            if not analysis.methodology_rationale:
                analysis.methodology_rationale = "O plano apresenta características de 'Americanidade' (exercício a qualquer momento após vesting) e barreiras complexas (Lock-up ou Correção de Strike). Black-Scholes assume exercício apenas no vencimento (Europeia) e Strike constante. O Binomial (Lattice) discretiza o tempo, permitindo verificar a otimalidade do exercício e aplicar o desconto de iliquidez (Lock-up) nó a nó."
            return analysis

        # Fallback padrão agora é BLACK_SCHOLES_GRADED, já que removemos o Vanilla
        analysis.model_recommended = PricingModelType.BLACK_SCHOLES_GRADED
        if not analysis.methodology_rationale:
            analysis.methodology_rationale = "O plano não apresenta características exóticas (barreiras complexas ou gatilhos de mercado). O Black-Scholes-Merton (Graded) é o padrão de mercado para precificação eficiente, calculando cada tranche de vesting como uma opção individual."
        
        return analysis

class DocumentService:
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

    @staticmethod
    def extract_text(uploaded_file) -> str:
        if not DocumentService.HAS_DOC_LIBS: return ""
        text = ""
        filename = uploaded_file.name.lower()
        try:
            if filename.endswith('.pdf'):
                reader = DocumentService.PyPDF2.PdfReader(uploaded_file)
                for page in reader.pages:
                    if page.extract_text(): text += page.extract_text()
            elif filename.endswith('.docx'):
                doc = DocumentService.Document(uploaded_file)
                text = "\n".join([p.text for p in doc.paragraphs])
        except Exception as e:
            st.error(f"Erro na leitura: {e}")
        return text

    @staticmethod
    @st.cache_data(show_spinner=False)
    def analyze_plan_with_gemini(text: str, api_key: str) -> Optional[PlanAnalysisResult]:
        if not DocumentService.HAS_GEMINI or not api_key:
            return None
            
        DocumentService.genai.configure(api_key=api_key)
        model = DocumentService.genai.GenerativeModel('gemini-2.5-flash')
        
        prompt = f"""
        Você é um Especialista em IFRS 2 e Atuária. Analise o contrato de ILP abaixo.
        Extraia parâmetros técnicos e forneça uma justificativa metodológica robusta.
        
        ATENÇÃO AOS DETALHES TÉCNICOS:
        - Vesting: Gradual? Performance?
        - Strike: Fixo ou indexado (IGPM/IPCA)?
        - Lock-up: Restrição de liquidez pós-exercício?
        - Gatilhos de Mercado: TSR, Valorização da Ação?
        - TIPO DE PLANO: Identifique se é Matching Shares ou Ações Restritas (Strike Zero).
        
        TEXTO:
        {text[:35000]}

        SAÍDA JSON (ESTRITA):
        {{
            "contract_features": "Texto resumindo as cláusulas críticas que afetam o preço (ex: 'Plano de Matching Shares com vesting em 3 tranches, Strike R$ 0,00, Lock-up de 2 anos'). Seja específico.",
            "summary": "Resumo geral do plano.",
            "model_data": {{
                "recommended_model": "Monte Carlo" | "Binomial" | "Black-Scholes" | "RSU",
                "deep_rationale": "Explicação técnica detalhada (2-3 parágrafos) justificando a escolha. Compare com as limitações dos outros modelos para este caso específico. Ex: 'Por se tratar de Matching Shares (Strike Zero), o modelo RSU é o mais indicado...'",
                "justification": "Frase curta para UI.",
                "comparison": "Comparação breve.",
                "pros": ["Pró 1", "Pró 2"], 
                "cons": ["Contra 1", "Contra 2"],
                "params": {{
                    "option_life": <float>,
                    "strike_price": <float>,
                    "strike_is_zero": <bool, IMPORTANTE: true para Matching Shares>,
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
            
            # Mapeamento
            rec_model_str = data.get('model_data', {}).get('recommended_model', '')
            model_enum = PricingModelType.UNDEFINED
            for m in PricingModelType:
                # Ajuste no mapeamento para ignorar 'Vanilla' se vier da IA e mapear para Graded
                if m.name in rec_model_str.upper().replace(" ", "_"):
                    model_enum = m
                    break
            
            # Fallback se IA sugerir BS Vanilla antigo
            if model_enum == PricingModelType.UNDEFINED and "BLACK" in rec_model_str.upper():
                model_enum = PricingModelType.BLACK_SCHOLES_GRADED

            params = data.get('model_data', {}).get('params', {})
            
            tranches = []
            for t in params.get('vesting_schedule', []):
                try:
                    p_y = float(t.get('period_years', 0))
                    p_p = float(t.get('percentage', 0))
                    if p_y > 0: tranches.append(Tranche(p_y, p_p))
                except (ValueError, TypeError):
                    continue

            # Função auxiliar para converter com fallback seguro
            def safe_float(val, default=0.0):
                try:
                    return float(val)
                except (ValueError, TypeError):
                    return default

            return PlanAnalysisResult(
                summary=data.get('summary', ''),
                contract_features=data.get('contract_features', 'Características não identificadas.'),
                methodology_rationale=data.get('model_data', {}).get('deep_rationale', ''),
                model_recommended=model_enum,
                model_reason=data.get('model_data', {}).get('justification', ''),
                model_comparison=data.get('model_data', {}).get('comparison', ''),
                pros=data.get('model_data', {}).get('pros', []),
                cons=data.get('model_data', {}).get('cons', []),
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
        except Exception as e:
            st.error(f"Erro IA: {e}")
            return DocumentService.mock_analysis(text)

    # --- MÉTODO PARA GERAÇÃO DE CÓDIGO ---
    @staticmethod
    def generate_custom_monte_carlo_code(contract_text: str, params: Dict, api_key: str) -> str:
        """Gera código Python customizado via IA para Monte Carlo complexo."""
        if not api_key: return "# API Key necessária para gerar código customizado."
        
        DocumentService.genai.configure(api_key=api_key)
        model = DocumentService.genai.GenerativeModel('gemini-2.5-flash')
        
        # Converte valores numéricos para string segura no JSON
        safe_params = {k: (float(v) if isinstance(v, (int, float, np.number)) else v) for k, v in params.items()}
        
        prompt = f"""
        Você é um Engenheiro Financeiro Quantitativo (Python Expert).
        
        TAREFA:
        Escreva um script Python COMPLETO e EXECUTÁVEL para precificar o plano de opções descrito abaixo usando Simulação de Monte Carlo.
        
        REQUISITOS OBRIGATÓRIOS:
        1. Capture particularidades: Vesting por Performance, Gatilhos de TSR, Barreiras, Lock-up.
        2. Use `numpy` para vetorização.
        3. Defina variáveis no início do script com base nos PARÂMETROS BASE fornecidos.
        4. O código deve imprimir o Fair Value final e o Erro Padrão com `print()`.
        5. NÃO use placeholders. O código deve rodar imediatamente se copiado.
        6. IMPORTANTE: Envolva a lógica principal em uma função chamada `run_simulation()`.
        7. CRÍTICO: No final do script, CHAME explicitamente `run_simulation()` e atribua o resultado à variável global `fv`.
           Exemplo:
           `fv = run_simulation()`
           `print(f'Resultado Final: {{fv}}')`
        8. CÁLCULO POR TRANCHE (Graded Vesting):
           O código DEVE iterar sobre a lista de datas de vesting (`vesting_schedule`) fornecida nos parâmetros base.
           O valor da opção deve ser calculado individualmente para cada tranche (considerando o tempo de vesting específico daquela tranche) e somado no final para obter o Fair Value total do plano.
           Não calcule como se fosse um único bloco de vencimento, a menos que haja apenas uma data.
        
        PARÂMETROS BASE:
        {json.dumps(safe_params, indent=2, default=str)}
        
        CONTRATO (Trecho):
        {contract_text[:15000]}
        
        SAÍDA: Apenas o bloco de código Python. Não inclua markdown ```python no início ou fim.
        """
        try:
            response = model.generate_content(prompt)
            code = response.text
            # Limpeza extra para garantir que não venha markdown
            code = re.sub(r'^```python\s*', '', code)
            code = re.sub(r'^```\s*', '', code)
            code = re.sub(r'\s*```$', '', code)
            return code.strip()
        except Exception as e:
            return f"# Erro ao gerar código: {e}"

    @staticmethod
    def mock_analysis(text: str) -> PlanAnalysisResult:
        tranches = [Tranche(1.0, 0.25), Tranche(2.0, 0.25), Tranche(3.0, 0.25), Tranche(4.0, 0.25)]
        return PlanAnalysisResult(
            summary="[MOCK] Plano simulado: 4 tranches, correção de strike (IGPM), lock-up 2 anos.",
            contract_features="[MOCK] Características Principais:\n- Vesting em 4 tranches anuais (25% cada).\n- Correção do Strike pelo IGPM.\n- Lock-up de 2 anos após exercício.\n- Liquidação física em ações.",
            methodology_rationale="[MOCK] A recomendação pelo Modelo Binomial se justifica pela presença de Lock-up (restrição de venda) e correção monetária do Strike. O modelo Black-Scholes (Vanilla) assume exercício apenas no vencimento ou constante, e não captura a desvalorização da opção causada pela falta de liquidez (Lock-up), nem a variação do Strike no tempo. O Binomial permite modelar a árvore de decisão considerando essas barreiras dinamicamente.",
            model_recommended=PricingModelType.BINOMIAL, 
            model_reason="Detectado Strike Dinâmico e Lock-up. Requer Binomial.",
            model_comparison="Binomial trata inflação e lock-up melhor que BS.",
            pros=["Trata Lock-up", "Inflação"], cons=["Complexidade"],
            tranches=tranches,
            has_strike_correction=True,
            option_life_years=10.0,
            strike_price=10.0,
            lockup_years=2.0,
            turnover_rate=0.05,
            early_exercise_multiple=2.5
        )

# =============================================================================
# CAMADA 4: INTERFACE (UI)
# =============================================================================

class IFRS2App:
    def run(self):
        st.title("🛡️ Icarus: Validador IFRS 2 (Completo)")
        
        # Inicializa estado da sessão para persistência da análise e tranches
        if 'analysis_result' not in st.session_state:
            st.session_state['analysis_result'] = None
        if 'full_context_text' not in st.session_state:
            st.session_state['full_context_text'] = ""
        if 'tranches' not in st.session_state:
            st.session_state['tranches'] = []

        with st.sidebar:
            st.header("Entradas")
            # --- MODIFICAÇÃO PARA USAR SECRETS ---
            if "GEMINI_API_KEY" in st.secrets:
                gemini_key = st.secrets["GEMINI_API_KEY"]
                st.success("🔑 API Key detectada (Secrets)")
            else:
                gemini_key = st.text_input("Gemini API Key", type="password")
            
            st.subheader("Dados do Plano")
            # Upload de arquivo (MODIFICADO: aceita múltiplos arquivos)
            uploaded_files = st.file_uploader("1. Upload de Contratos (Opcional)", type=['pdf', 'docx'], accept_multiple_files=True)
            
            # Campo para inserção manual
            manual_text = st.text_area(
                "2. Características Manuais (Opcional)", 
                height=200, 
                placeholder="Cole trechos do contrato ou descreva as regras:\nEx: 'Vesting em 4 anos, Strike R$ 10,00 corrigido por IGPM...'"
            )
            
            st.text_input("Ticker", "VALE3")
            
            # Botão de processamento para combinar as entradas
            if st.button("🚀 Analisar Plano", type="primary"):
                combined_text = ""
                
                # Extrai texto dos arquivos se existirem
                if uploaded_files:
                    with st.spinner("Lendo arquivos..."):
                        for f in uploaded_files:
                            combined_text += f"--- INÍCIO DO ARQUIVO: {f.name} ---\n"
                            combined_text += DocumentService.extract_text(f) + "\n"
                            combined_text += f"--- FIM DO ARQUIVO: {f.name} ---\n\n"
                
                # Adiciona texto manual se existir
                if manual_text:
                    combined_text += f"--- DADOS/REGRAS INSERIDOS MANUALMENTE ---\n{manual_text}"
                
                # Validação: Pelo menos um dos dois deve existir
                if not combined_text.strip():
                    st.error("⚠️ É necessário subir um arquivo OU digitar as características manualmente para prosseguir.")
                else:
                    st.session_state['full_context_text'] = combined_text
                    
                    if gemini_key:
                        with st.spinner("🤖 IA Analisando estrutura do plano..."):
                            analysis = DocumentService.analyze_plan_with_gemini(combined_text, gemini_key)
                    else:
                        st.warning("⚠️ Sem API Key: Usando dados Mockados para demonstração.")
                        analysis = DocumentService.mock_analysis(combined_text)
                    
                    if analysis:
                        # Executa seletor de modelo
                        analysis = ModelSelectorService.select_model(analysis)
                        st.session_state['analysis_result'] = analysis
                        
                        # Inicializa as tranches editáveis com o que veio da análise ou um default
                        if analysis.tranches:
                            st.session_state['tranches'] = [t for t in analysis.tranches]
                        else:
                            st.session_state['tranches'] = [Tranche(1.0, 1.0)]
            
        # Exibição Principal (Baseada no estado da sessão)
        if st.session_state['analysis_result']:
            self._render_dashboard(
                st.session_state['analysis_result'], 
                st.session_state['full_context_text'], 
                gemini_key
            )
        else:
            # Estado inicial
            st.info("👈 Por favor, forneça o contrato (upload) ou a descrição do plano (texto) na barra lateral para iniciar a análise.")

    def _render_dashboard(self, analysis: PlanAnalysisResult, full_text: str, api_key: str):
        st.subheader("1. Diagnóstico e Seleção de Modelo")
        
        # LAYOUT RENOVADO: Exibição detalhada das características e racional
        with st.container():
            col_diag_1, col_diag_2 = st.columns([1, 1])
            
            with col_diag_1:
                st.markdown("### 📋 Características do Contrato")
                if analysis.contract_features:
                    st.info(analysis.contract_features)
                else:
                    st.info(analysis.summary) # Fallback

            with col_diag_2:
                st.markdown("### 🧠 Racional Metodológico")
                st.success(f"**Modelo Recomendado:** {analysis.model_recommended.value}")
                
                if analysis.methodology_rationale:
                     st.write(analysis.methodology_rationale)
                else:
                     st.write(analysis.model_reason) # Fallback

        with st.expander("Ver Comparativo de Modelos", expanded=False):
             st.write(analysis.model_comparison)
             c_pros, c_cons = st.columns(2)
             c_pros.write("**Prós:**")
             for p in analysis.pros: c_pros.write(f"- {p}")
             c_cons.write("**Contras:**")
             for c in analysis.cons: c_cons.write(f"- {c}")

        st.divider()

        opts = [m for m in PricingModelType if m != PricingModelType.UNDEFINED]
        try: idx = opts.index(analysis.model_recommended)
        except: idx = 0
        active_model = st.selectbox("Modelo Ativo (Cálculo):", opts, index=idx)
        
        st.divider()

        st.subheader("2. Parâmetros de Mercado (Base)")
        c1, c2, c3, c4 = st.columns(4)
        S = c1.number_input("Preço Spot (R$)", 0.0, 5000.0, 50.0)
        K = c2.number_input("Strike (R$)", 0.0, 5000.0, analysis.strike_price)
        vol = c3.number_input("Volatilidade (%)", 0.0, 200.0, 30.0) / 100
        r = c4.number_input("Taxa Livre Risco (%)", 0.0, 50.0, 10.75) / 100
        q_global = st.number_input("Dividend Yield (% a.a.)", 0.0, 20.0, 4.0) / 100

        st.subheader("3. Cálculo do Fair Value")

        if active_model == PricingModelType.BLACK_SCHOLES_GRADED:
            self._render_graded(S, K, r, vol, q_global, analysis)
        elif active_model == PricingModelType.BINOMIAL:
            self._render_binomial_graded(S, K, r, vol, q_global, analysis)
        elif active_model == PricingModelType.MONTE_CARLO:
            self._render_monte_carlo_ai(S, K, r, vol, q_global, analysis, full_text, api_key)
        elif active_model == PricingModelType.RSU:
            self._render_rsu(S, r, q_global, analysis)
        else:
            st.warning("Selecione um modelo válido.")

    def _manage_tranches(self):
        """Helper para adicionar/remover tranches na interface."""
        st.markdown("#### ⚙️ Gerenciar Tranches")
        c1, c2 = st.columns(2)
        if c1.button("➕ Adicionar Tranche"):
            # Adiciona uma nova tranche padrão (1 ano a mais que a última ou 1.0 se vazia)
            last_vesting = st.session_state['tranches'][-1].vesting_date if st.session_state['tranches'] else 0.0
            st.session_state['tranches'].append(Tranche(last_vesting + 1.0, 0.0))
            st.rerun() # Atualiza a interface
            
        if c2.button("➖ Remover Última"):
            if len(st.session_state['tranches']) > 0:
                st.session_state['tranches'].pop()
                st.rerun()

    def _render_rsu(self, S, r, q, analysis):
        st.info("ℹ️ Valuation de RSU / Matching Shares (Por Tranche)")
        st.caption("Cálculo: Preço da Ação descontado de Dividendos e Lock-up para cada período de vesting.")
        
        self._manage_tranches()
        
        tranches = st.session_state['tranches']
        tranche_inputs = []
        
        if not tranches:
            st.warning("Nenhuma tranche definida. Adicione tranches acima.")
            return

        for i, t in enumerate(tranches):
            with st.expander(f"Tranche {i+1}", expanded=True):
                c1, c2, c3, c4 = st.columns(4)
                t_vest = c1.number_input(f"Vesting (Anos) {i}", value=float(t.vesting_date), key=f"rsu_v_{i}")
                t_lock = c2.number_input(f"Lock-up (Anos) {i}", value=float(analysis.lockup_years), key=f"rsu_l_{i}")
                t_vol = c3.number_input(f"Volatilidade % (Lockup) {i}", value=30.0, key=f"rsu_vol_{i}") / 100
                t_prop = c4.number_input(f"Proporção % {i}", value=float(t.proportion * 100), key=f"rsu_prop_{i}") / 100
                
                tranche_inputs.append({
                    "T": t_vest, "lockup": t_lock, "vol": t_vol, "prop": t_prop
                })

        st.divider()
        if st.button("Calcular Fair Value (RSU)"):
            total_fv = 0.0
            res_data = []
            
            for i, inp in enumerate(tranche_inputs):
                # Base Value: S * exp(-q * T)
                base_fv = S * np.exp(-q * inp["T"])
                
                # Lockup Discount (Chaffe)
                discount = 0.0
                if inp["lockup"] > 0:
                    discount = FinancialMath.calculate_lockup_discount(inp["vol"], inp["lockup"], base_fv, q)
                
                unit_fv = base_fv - discount
                weighted_fv = unit_fv * inp["prop"]
                total_fv += weighted_fv
                
                res_data.append({
                    "Tranche": i+1, 
                    "Vesting": inp["T"], 
                    "Valor Base": base_fv, 
                    "Desconto": discount, 
                    "FV Unitário": unit_fv,
                    "FV Ponderado": weighted_fv
                })
            
            c_res1, c_res2 = st.columns([1, 2])
            c_res1.metric("Fair Value Total (Ponderado)", f"R$ {total_fv:.4f}")
            c_res2.dataframe(pd.DataFrame(res_data))

    def _render_graded(self, S, K, r, vol, q, analysis):
        st.info("ℹ️ Cálculo por Tranche Individual (Black-Scholes)")
        st.caption("Ajuste as premissas para cada tranche individualmente.")
        
        self._manage_tranches()
        
        tranches = st.session_state['tranches']
        tranche_inputs = []
        
        if not tranches:
            st.warning("Nenhuma tranche definida.")
            return
        
        for i, t in enumerate(tranches):
            with st.expander(f"Tranche {i+1}: Vesting Original {t.vesting_date} anos", expanded=True):
                c_s, c_k, c_t = st.columns(3)
                t_s = c_s.number_input(f"Spot ({i})", value=float(S), format="%.2f", key=f"bs_s_{i}")
                t_k = c_k.number_input(f"Strike ({i})", value=float(K), format="%.2f", key=f"bs_k_{i}")
                t_date = c_t.number_input(f"Vesting ({i})", value=float(t.vesting_date), format="%.2f", key=f"bs_t_{i}")
                
                c_r, c_v, c_q = st.columns(3)
                t_r = c_r.number_input(f"Taxa ({i})", value=float(r), format="%.4f", key=f"bs_r_{i}")
                t_v = c_v.number_input(f"Vol ({i})", value=float(vol), format="%.4f", key=f"bs_v_{i}")
                t_q = c_q.number_input(f"Div Yield ({i})", value=float(q), format="%.4f", key=f"bs_q_{i}")
                
                # Input de proporção para ponderação
                t_prop = st.number_input(f"Peso da Tranche (%) {i}", value=float(t.proportion * 100), key=f"bs_p_{i}") / 100
                
                tranche_inputs.append({
                    "S": t_s, "K": t_k, "T": t_date, "r": t_r, "vol": t_v, "q": t_q, "prop": t_prop
                })

        st.divider()
        if st.button("Calcular Fair Value Total (Black-Scholes Graded)", type="primary"):
            total_fv = 0.0
            res_data = []
            for i, inp in enumerate(tranche_inputs):
                fv = FinancialMath.bs_call(inp["S"], inp["K"], inp["T"], inp["r"], inp["vol"], inp["q"])
                w_fv = fv * inp["prop"]
                total_fv += w_fv
                res_data.append({"Tranche": i+1, "Vesting": inp["T"], "FV Unit": fv, "FV Ponderado": w_fv})
            
            c_res1, c_res2 = st.columns([1, 2])
            c_res1.metric("Fair Value Total", f"R$ {total_fv:.4f}")
            c_res2.dataframe(pd.DataFrame(res_data))

    def _render_binomial_graded(self, S, K, r, vol, q, analysis):
        st.info("ℹ️ Binomial por Tranche (Vesting Escalonado)")
        st.caption("Calcula uma árvore binomial separada para cada período de vesting. Todos os parâmetros são editáveis por tranche.")
        
        self._manage_tranches()
        tranches = st.session_state['tranches']
        tranche_inputs = []
        
        if not tranches:
            st.warning("Nenhuma tranche definida.")
            return
        
        for i, t in enumerate(tranches):
            with st.expander(f"Tranche {i+1}: Vesting {t.vesting_date} anos", expanded=True):
                c1, c2, c3 = st.columns(3)
                t_s = c1.number_input(f"Spot ({i})", value=float(S), format="%.2f", key=f"bin_s_{i}")
                t_k = c2.number_input(f"Strike ({i})", value=float(K), format="%.2f", key=f"bin_k_{i}")
                t_vest = c3.number_input(f"Vesting ({i})", value=float(t.vesting_date), format="%.2f", key=f"bin_vest_{i}")
                
                c4, c5, c6 = st.columns(3)
                t_r = c4.number_input(f"Taxa ({i})", value=float(r), format="%.4f", key=f"bin_r_{i}")
                t_v = c5.number_input(f"Vol ({i})", value=float(vol), format="%.4f", key=f"bin_v_{i}")
                t_q = c6.number_input(f"Div Yld ({i})", value=float(q), format="%.4f", key=f"bin_q_{i}")

                c7, c8, c9 = st.columns(3)
                t_life = c7.number_input(f"Vida Total ({i})", value=float(analysis.option_life_years), key=f"bin_life_{i}")
                t_infl = c8.number_input(f"Corr. Strike % ({i})", value=4.5 if analysis.has_strike_correction else 0.0, key=f"bin_inf_{i}") / 100
                t_lock = c9.number_input(f"Lock-up ({i})", value=float(analysis.lockup_years), key=f"bin_lck_{i}")
                
                c10, c11, c12 = st.columns(3)
                t_w = c10.number_input(f"Turnover ({i})", value=float(analysis.turnover_rate), format="%.3f", key=f"bin_w_{i}")
                t_m = c11.number_input(f"Múltiplo M ({i})", value=float(analysis.early_exercise_multiple), key=f"bin_m_{i}")
                t_h = c12.number_input(f"Hurdle ({i})", value=0.0, key=f"bin_h_{i}")
                
                t_prop = st.number_input(f"Peso % {i}", value=float(t.proportion * 100), key=f"bin_prop_{i}") / 100

                tranche_inputs.append({
                    "S": t_s, "K": t_k, "r": t_r, "vol": t_v, "q": t_q,
                    "vesting": t_vest, "T_life": t_life, "infl": t_infl, "lockup": t_lock,
                    "w": t_w, "m": t_m, "hurdle": t_h, "tipo": 0, "prop": t_prop
                })
        
        st.divider()
        self._render_binomial_calculation_logic(tranche_inputs)

    def _render_binomial_calculation_logic(self, inputs_list):
        if st.button("Calcular Fair Value Total (Binomial Graded)", type="primary"):
            total_fv = 0.0
            res_data = []
            progress_bar = st.progress(0)
            
            for i, inp in enumerate(inputs_list):
                fv = FinancialMath.binomial_custom_optimized(
                    inp["S"], inp["K"], inp["r"], inp["vol"], inp["q"],
                    inp["vesting"], inp["w"], inp["m"], inp["hurdle"],
                    inp["T_life"], inp["infl"], inp["lockup"], inp["tipo"]
                )
                w_fv = fv * inp["prop"]
                total_fv += w_fv
                res_data.append({"Tranche": i+1, "Vesting": inp["vesting"], "FV Unit": fv, "FV Ponderado": w_fv})
                progress_bar.progress((i + 1) / len(inputs_list))
            
            c_res1, c_res2 = st.columns([1, 2])
            c_res1.metric("Fair Value Total", f"R$ {total_fv:.4f}")
            c_res2.dataframe(pd.DataFrame(res_data))

    def _render_monte_carlo_ai(self, S, K, r, vol, q, analysis, text, api_key):
        st.warning("⚠️ Monte Carlo: Gerador de Código IA Ativado.")
        
        # 1. Sugestão de Parâmetros pela IA
        if 'mc_code' not in st.session_state:
            st.session_state['mc_code'] = ""
            
        # Pega as tranches editadas pelo usuário (se houver) ou do analysis
        current_tranches = st.session_state.get('tranches', analysis.tranches)
        vesting_dates = [t.vesting_date for t in current_tranches] if current_tranches else [1.0, 2.0, 3.0]

        params = {
            "S0": S, "K": K, "r": r, "sigma": vol, "q": q,
            "T": analysis.option_life_years,
            "vesting_schedule": vesting_dates
        }
        
        col_gen, col_exec = st.columns(2)
        
        # Passo 1: Gerar Código
        if col_gen.button("1. Gerar Código Customizado"):
            with st.spinner("Escrevendo script de simulação..."):
                code = DocumentService.generate_custom_monte_carlo_code(text, params, api_key)
                st.session_state['mc_code'] = code
        
        # Passo 2: Mostrar e Permitir Edição
        if st.session_state['mc_code']:
            st.markdown("### 🐍 Script de Simulação (Editável)")
            st.caption("Verifique os parâmetros abaixo antes de executar. Você pode ajustar `n_sims`, `S0`, etc. diretamente no código.")
            
            edited_code = st.text_area("Código Python", value=st.session_state['mc_code'], height=400)
            st.session_state['mc_code'] = edited_code # Atualiza com edição manual
            
            # Passo 3: Executar
            if col_exec.button("2. Executar Modelo", type="primary"):
                with st.spinner("Rodando simulação de Monte Carlo..."):
                    # Segurança: Executar código arbitrário é perigoso em prod. 
                    # Aqui rodamos em um ambiente local restrito (exec) apenas para demonstração.
                    # Redirecionar stdout para capturar prints
                    old_stdout = io.StringIO()
                    try:
                        # Contexto local seguro
                        local_scope = {}
                        with st.spinner("Computando..."):
                            # Captura output
                            import sys
                            original_stdout = sys.stdout
                            sys.stdout = old_stdout
                            
                            # --- EXECUÇÃO DO CÓDIGO ---
                            # IMPORTANTE: Passar local_scope como SEGUNDO argumento para 
                            # que funcione como globals e locals ao mesmo tempo.
                            # Isso corrige o erro "name 'vesting_schedule' is not defined".
                            exec(edited_code, local_scope)
                            
                            # --- CORREÇÃO DE SEGURANÇA ---
                            # Se a IA definiu a função run_simulation mas não a chamou explicitamente (comum),
                            # ou se chamou mas não guardou em 'fv', tentamos corrigir aqui.
                            if 'fv' not in local_scope:
                                if 'run_simulation' in local_scope:
                                    # Força execução da função
                                    ret = local_scope['run_simulation']()
                                    if ret is not None and isinstance(ret, (float, int, np.number)):
                                        local_scope['fv'] = ret
                                        print(f"Resultado (Forçado): {ret}")
                            
                            sys.stdout = original_stdout
                            output = old_stdout.getvalue()
                            
                        st.success("Simulação Concluída!")
                        st.markdown("### Resultados")
                        st.text(output)
                        
                        # Tentar extrair FV do escopo se definido
                        if 'fv' in local_scope:
                            st.metric("Fair Value (Variável 'fv')", f"R$ {local_scope['fv']:.4f}")
                        elif not output.strip():
                            st.warning("O script rodou mas não gerou saída visual (print) nem variável 'fv'. Verifique se a função principal foi chamada no código.")
                            
                    except Exception as e:
                        st.error(f"Erro na execução do código: {e}")

    # Método vanilla removido pois foi depreciado

if __name__ == "__main__":
    app = IFRS2App()
    app.run()
