import io
from datetime import date, timedelta
import pandas as pd
from docxtpl import DocxTemplate
import streamlit as st

class ReportService:
    """
    Serviço de Geração de Laudos Contábeis (Docx).
    Mapeia inputs da interface + resultados do cálculo para o Template.
    """

    @staticmethod
    def _format_currency(value):
        try:
            return f"R$ {float(value):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        except:
            return str(value)

    @staticmethod
    def _format_date(dt):
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
    def generate_report_context(
        analysis_result, 
        tranches, 
        calc_results, 
        manual_inputs
    ) -> dict:
        """
        Constrói o dicionário 'context' mesclando dados automáticos e manuais.
        """
        emp_info = manual_inputs.get('empresa', {})
        prog_info = manual_inputs.get('programa', {})
        resp_info = manual_inputs.get('responsavel', {})
        contab_info = manual_inputs.get('contab', {})

        # 1. Estrutura Base (Tags simples)
        context = {
            "empresa": {
                "nome": emp_info.get("nome", "EMPRESA N/A"),
                "ticker": emp_info.get("ticker", ""),
                "capital_aberto": emp_info.get("capital_aberto", False),
            },
            "programa": {
                "nome": prog_info.get("nome", "Plano de Incentivo"),
                "tipo_descritivo": "Plano Baseado em Ações",
                "qtd_beneficiarios": prog_info.get("qtd_beneficiarios", 1),
                "data_outorga": ReportService._format_date(prog_info.get("data_outorga", date.today())),
                "forma_liquidacao": "INSTRUMENTOS DE PATRIMÔNIO" if "EQUITY" in str(analysis_result.settlement_type) else "CAIXA",
                "metodologia": analysis_result.model_recommended.name,
            },
            "laudo": {
                "data_extenso": ReportService._get_data_extenso(date.today()),
                "arquivo_excel": "Anexo I - Memória de Cálculo.xlsx"
            },
            "responsavel": {
                "nome": resp_info.get("nome", ""),
                "cargo": resp_info.get("cargo", ""),
                "email": resp_info.get("email", "")
            },
            "regras": {
                "texto_vesting": f"escalonado em {len(tranches)} tranches",
                "tem_performance": analysis_result.has_market_condition,
                "texto_performance": "condições de mercado (TSR)" if analysis_result.has_market_condition else "apenas tempo de serviço",
                "tem_lockup": analysis_result.lockup_years > 0,
                "prazo_lockup": f"{analysis_result.lockup_years} anos",
                "tem_performance_nao_mercado": contab_info.get("tem_metas_nao_mercado", False)
            },
            "calculo": {
                "data_base": ReportService._format_date(prog_info.get("data_outorga", date.today())),
                "valor_ativo_base": ReportService._format_currency(calc_results[0].get('S', 0) if calc_results else 0),
                "bolsa": "B3 S.A.",
                "tem_correcao_strike": analysis_result.has_strike_correction,
                "indice_correcao": "IGPM/IPCA", 
                "modelo_precificacao": analysis_result.model_recommended.name,
                "multiplo_exercicio": analysis_result.early_exercise_multiple,
                "taxa_turnover_pos": f"{analysis_result.turnover_rate*100:.1f}%",
                "dividend_yield": f"{calc_results[0].get('q', 0)*100:.1f}%" if calc_results else "0,0%"
            },
            "contab": {
                "taxa_turnover": f"{contab_info.get('taxa_turnover', 0)*100:.1f}%",
                "percentual_atingimento": f"{contab_info.get('percentual_atingimento', 100):.1f}%",
                "tem_encargos": contab_info.get("tem_encargos", False)
            },
            "tabelas": {}
        }

        # 2. Tabelas Dinâmicas
        tbl_cronograma = []
        tbl_resultados = []
        tbl_encargos = []
        
        data_outorga = prog_info.get("data_outorga", date.today())
        
        # Reconstrói dados baseados nos resultados do cálculo
        for i, row in enumerate(calc_results):
            lote_nome = f"Lote {row.get('Tranche', i+1)}"
            
            # Tenta pegar vesting/vencimento do resultado ou da tranche original
            vesting_val = float(row.get('Vesting', 0))
            if vesting_val == 0 and i < len(tranches):
                vesting_val = tranches[i].vesting_date
                
            dt_vesting = data_outorga + timedelta(days=int(vesting_val*365))
            # Estimativa simples de vencimento se não vier no row
            t_life = float(row.get('T', 5.0))
            dt_venc = data_outorga + timedelta(days=int(t_life*365))

            tbl_cronograma.append({
                "nome": context['programa']['nome'],
                "numero": lote_nome,
                "qtd": "N/D", # Dado não vem do valuation puramente financeiro
                "data_outorga": ReportService._format_date(data_outorga),
                "data_vesting": ReportService._format_date(dt_vesting),
                "data_vencimento": ReportService._format_date(dt_venc)
            })

            tbl_resultados.append({
                "lote": lote_nome,
                "modelo": context['programa']['metodologia'],
                "fv_final": ReportService._format_currency(row.get('FV Unit', 0))
            })

        # Encargos (Se selecionado)
        if context['contab']['tem_encargos']:
            tbl_encargos.append({"nome": "INSS Patronal + RAT + Terceiros", "valor": "28,0%"})
            tbl_encargos.append({"nome": "FGTS", "valor": "8,0%"})

        # Projeção de Despesas (Mock Simples Linear)
        tbl_projecao = []
        total_fv = sum([float(r.get('FV Ponderado', 0)) for r in calc_results])
        anos_projecao = 3
        custo_anual = total_fv / anos_projecao
        
        for ano in range(anos_projecao):
            ano_corrente = data_outorga.year + ano
            encargo_val = (custo_anual * 0.36) if context['contab']['tem_encargos'] else 0.0
            tbl_projecao.append({
                "ano": ano_corrente,
                "custo_ilp": ReportService._format_currency(custo_anual),
                "custo_encargos": ReportService._format_currency(encargo_val),
                "total": ReportService._format_currency(custo_anual + encargo_val)
            })

        context["tabelas"]["cronograma"] = tbl_cronograma
        context["tabelas"]["resultados_fair_value"] = tbl_resultados
        context["tabelas"]["encargos"] = tbl_encargos
        context["tabelas"]["projecao_despesas"] = tbl_projecao
        
        # Tabelas vazias para evitar erro no template se não usadas
        if "volatilidade" not in context["tabelas"]: context["tabelas"]["volatilidade"] = []
        if "taxa_livre_risco" not in context["tabelas"]: context["tabelas"]["taxa_livre_risco"] = []
        if "strikes" not in context["tabelas"]: context["tabelas"]["strikes"] = []

        return context

    @staticmethod
    def render_template(template_file, context) -> io.BytesIO:
        doc = DocxTemplate(template_file)
        doc.render(context)
        output = io.BytesIO()
        doc.save(output)
        output.seek(0)
        return output
