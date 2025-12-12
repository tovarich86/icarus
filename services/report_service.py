import io
from datetime import date, timedelta
import pandas as pd
from docxtpl import DocxTemplate
import locale

# Tenta configurar locale para PT-BR (opcional, fallback manual)
try:
    locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')
except:
    pass

class ReportService:
    """
    Serviço de Geração de Laudos Contábeis (Docx).
    Mapeia inputs da interface + resultados do cálculo para o Template Jinja2.
    """

    @staticmethod
    def _format_currency(value):
        """Formata moeda R$ #.##0,00"""
        try:
            val = float(value)
            return f"R$ {val:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        except:
            return str(value)

    @staticmethod
    def _format_percent(value):
        """Formata percentual #.##0,00%"""
        try:
            val = float(value) * 100
            return f"{val:,.2f}%".replace(".", ",")
        except:
            return str(value)

    @staticmethod
    def _format_date(dt):
        """Formata data DD/MM/AAAA"""
        if isinstance(dt, (date, pd.Timestamp)):
            return dt.strftime("%d/%m/%Y")
        return str(dt)

    @staticmethod
    def _get_data_extenso(dt):
        meses = {
            1: 'janeiro', 2: 'fevereiro', 3: 'março', 4: 'abril',
            5: 'maio', 6: 'junho', 7: 'julho', 8: 'agosto',
            9: 'setembro', 10: 'outubro', 11: 'novembro', 12: 'dezembro'
        }
        return f"{dt.day} de {meses.get(dt.month, '')} de {dt.year}"

    @staticmethod
    def generate_report_context(analysis_result, tranches, calc_results, manual_inputs) -> dict:
        
        emp_info = manual_inputs.get('empresa', {})
        prog_info = manual_inputs.get('programa', {})
        resp_info = manual_inputs.get('responsavel', {})
        contab_info = manual_inputs.get('contab', {})
        extra_info = manual_inputs.get('calculo_extra', {})

        data_outorga = prog_info.get("data_outorga", date.today())
        
        # Mapeamento Completo
        context = {
            "empresa": {
                "nome": emp_info.get("nome", "EMPRESA N/A"),
                "ticker": emp_info.get("ticker", ""),
                "capital_aberto": emp_info.get("capital_aberto", False),
            },
            "programa": {
                "nome": prog_info.get("nome", "Plano de Incentivo"),
                "tipo_descritivo": "Plano Baseado em Ações",
                "tipo_detalhado": prog_info.get("tipo_detalhado", "Plano de Opção de Compra"),
                "qtd_beneficiarios": prog_info.get("qtd_beneficiarios", 1),
                "data_outorga": ReportService._format_date(data_outorga),
                "forma_liquidacao": prog_info.get("forma_liquidacao", "CAIXA"),
                "metodologia": prog_info.get("metodologia", "BLACK_SCHOLES"),
            },
            "laudo": {
                "data_extenso": ReportService._get_data_extenso(date.today()),
                "arquivo_excel": "Anexo I - Memória de Cálculo.xlsx"
            },
            "responsavel": resp_info,
            "regras": {
                "texto_vesting": f"escalonado em {len(tranches)} tranches",
                "tem_performance": analysis_result.has_market_condition,
                "texto_performance": "condições de mercado (TSR)" if analysis_result.has_market_condition else "tempo de serviço",
                "tem_lockup": analysis_result.lockup_years > 0,
                "prazo_lockup": f"{analysis_result.lockup_years} anos",
                "tem_performance_nao_mercado": False 
            },
            "calculo": {
                "data_base": ReportService._format_date(data_outorga),
                "valor_ativo_base": ReportService._format_currency(calc_results[0].get('S', 0) if calc_results else 0),
                "bolsa": emp_info.get("bolsa_nome", "B3 S.A.") if emp_info.get("capital_aberto") else "N/A",
                "metodo_precificacao_privado": extra_info.get("metodo_privado", "Avaliação Interna"),
                
                # TAGS CRÍTICAS PARA LÓGICA DE TEXTO
                "moeda": extra_info.get("moeda_selecionada", "BRL"), # Controla DI vs T-Bond
                "cenario_dividendos": extra_info.get("cenario_dividendos", "ZERO"), # Controla texto de dividendos
                
                "tem_correcao_strike": analysis_result.has_strike_correction,
                "indice_correcao": extra_info.get("indice_correcao_nome", "IGPM/IPCA"),
                "modelo_precificacao": prog_info.get("metodologia", "BLACK_SCHOLES"),
                "multiplo_exercicio": analysis_result.early_exercise_multiple,
                "taxa_turnover_pos": f"{analysis_result.turnover_rate*100:.1f}%",
                "dividend_yield": ReportService._format_percent(calc_results[0].get('q', 0)) if calc_results else "0,0%"
            },
            "contab": {
                 "taxa_turnover": f"{contab_info.get('taxa_turnover', 0)*100:.1f}%",
                 "tem_encargos": contab_info.get("tem_encargos", False)
            },
            "tabelas": {
                "cronograma": [], "strikes": [], "volatilidade": [], 
                "taxa_livre_risco": [], "resultados_fair_value": [], 
                "encargos": [], "projecao_despesas": []
            }
        }
        
        # Loop Único de Tabelas (Preenche TODAS as listas e usa Quantidade correta)
        for i, row in enumerate(calc_results):
            lote_nome = f"Lote {row.get('TrancheID', i+1)}"
            
            S = float(row.get('S', 0))
            K = float(row.get('K', 0))
            Vol = float(row.get('Vol', 0))
            r = float(row.get('r', 0))
            T = float(row.get('T', 0))
            vesting_val = float(row.get('Vesting', 0))
            
            if vesting_val == 0 and i < len(tranches):
                vesting_val = tranches[i].vesting_date
                
            dt_vesting = data_outorga + timedelta(days=int(vesting_val*365))
            dt_venc = data_outorga + timedelta(days=int(T*365))

            # AQUI: Usa a quantidade real definida no input
            qtd_total = prog_info.get("qtd_beneficiarios", 1)
            qtd_display = str(qtd_total) 

            # A. Tabela Cronograma
            context["tabelas"]["cronograma"].append({
                "nome": context['programa']['nome'],
                "numero": lote_nome,
                "qtd": qtd_display,  
                "data_outorga": ReportService._format_date(data_outorga),
                "data_vesting": ReportService._format_date(dt_vesting),
                "data_vencimento": ReportService._format_date(dt_venc)
            })

            # B. Tabela Strikes
            context["tabelas"]["strikes"].append({
                "lote": lote_nome,
                "strike": ReportService._format_currency(K)
            })

            # C. Tabela Volatilidade
            context["tabelas"]["volatilidade"].append({
                "data_base": ReportService._format_date(data_outorga),
                "vencimento": ReportService._format_date(dt_venc),
                "valor": ReportService._format_percent(Vol)
            })

            # D. Tabela Taxa Livre de Risco
            context["tabelas"]["taxa_livre_risco"].append({
                "lote": lote_nome,
                "vencimento": ReportService._format_date(dt_venc),
                "taxa": ReportService._format_percent(r)
            })

            # E. Resultado Final
            context["tabelas"]["resultados_fair_value"].append({
                "lote": lote_nome,
                "modelo": context['programa']['metodologia'],
                "fv_final": ReportService._format_currency(row.get('FV Unit', 0))
            })

        # --- ENCARGOS E PROJEÇÃO ---
        
        # Tabela Encargos
        if context['contab']['tem_encargos']:
            context["tabelas"]["encargos"].append({"nome": "INSS Patronal + RAT + Terceiros", "valor": "28,0%"})
            context["tabelas"]["encargos"].append({"nome": "FGTS", "valor": "8,0%"})

        # Projeção Linear Simples
        total_fv = sum([float(r.get('FV Ponderado', 0)) for r in calc_results])
        anos_projecao = 3
        custo_anual = total_fv / anos_projecao if anos_projecao > 0 else 0
        
        for ano in range(anos_projecao):
            ano_corrente = data_outorga.year + ano
            encargo_val = (custo_anual * 0.36) if context['contab']['tem_encargos'] else 0.0
            
            context["tabelas"]["projecao_despesas"].append({
                "ano": ano_corrente,
                "custo_ilp": ReportService._format_currency(custo_anual),
                "custo_encargos": ReportService._format_currency(encargo_val),
                "total": ReportService._format_currency(custo_anual + encargo_val)
            })

        return context

    @staticmethod
    def render_template(template_file, context) -> io.BytesIO:
        """Renderiza o arquivo final."""
        doc = DocxTemplate(template_file)
        doc.render(context)
        output = io.BytesIO()
        doc.save(output)
        output.seek(0)
        return output
