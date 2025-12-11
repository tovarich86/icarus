import io
from datetime import date, timedelta
import pandas as pd
from docxtpl import DocxTemplate
import streamlit as st

class ReportService:
    """
    Serviço de Geração de Laudos Contábeis (Docx).
    Mapeia os resultados do Valuation para as tags Jinja2 do Template.
    """

    @staticmethod
    def _format_currency(value):
        try:
            return f"R$ {float(value):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        except:
            return value

    @staticmethod
    def _format_date(dt):
        if isinstance(dt, (date, pd.Timestamp)):
            return dt.strftime("%d/%m/%Y")
        return str(dt)

    @staticmethod
    def generate_report_context(
        analysis_result, 
        tranches, 
        calc_results, 
        company_info, 
        general_info
    ) -> dict:
        """
        Constrói o dicionário 'context' que será injetado no Template.
        """
        
        # 1. Dados da Empresa e Programa
        context = {
            "empresa": {
                "nome": company_info.get("nome", "Empresa N/A"),
                "ticker": company_info.get("ticker", "N/A"),
                "capital_aberto": company_info.get("capital_aberto", True),
            },
            "programa": {
                "nome": general_info.get("nome_plano", "Plano de Incentivo"),
                "tipo_descritivo": "Plano de Opção de Compra de Ações", # Exemplo
                "qtd_beneficiarios": general_info.get("qtd_beneficiarios", 1),
                "data_outorga": ReportService._format_date(general_info.get("data_outorga", date.today())),
                "forma_liquidacao": "INSTRUMENTOS DE PATRIMÔNIO" if "EQUITY" in str(analysis_result.settlement_type) else "CAIXA",
                "metodologia": analysis_result.model_recommended.name if analysis_result.model_recommended else "BLACK_SCHOLES",
            },
            "laudo": {
                "data_extenso": date.today().strftime("%d de %B de %Y"), # Requer locale configurado ou formatador manual
                "arquivo_excel": "Anexo I - Memória de Cálculo.xlsx"
            },
            "responsavel": {
                "nome": general_info.get("resp_nome", ""),
                "cargo": general_info.get("resp_cargo", ""),
                "email": general_info.get("resp_email", "")
            },
            "regras": {
                "texto_vesting": f"escalonado em {len(tranches)} tranches anuais",
                "tem_performance": analysis_result.has_market_condition,
                "texto_performance": "condições de mercado (TSR) e metas operacionais" if analysis_result.has_market_condition else "apenas tempo de serviço",
                "tem_lockup": analysis_result.lockup_years > 0,
                "prazo_lockup": f"{analysis_result.lockup_years} anos"
            },
            "calculo": {
                "data_base": ReportService._format_date(general_info.get("data_outorga", date.today())),
                "valor_ativo_base": ReportService._format_currency(calc_results[0].get('S', 0) if calc_results else 0),
                "bolsa": "B3 S.A. - Brasil, Bolsa, Balcão",
                "tem_correcao_strike": analysis_result.has_strike_correction,
                "indice_correcao": "IGPM/IPCA",
                "modelo_precificacao": analysis_result.model_recommended.name,
                "multiplo_exercicio": analysis_result.early_exercise_multiple,
                "taxa_turnover_pos": f"{analysis_result.turnover_rate*100:.1f}%",
                "dividend_yield": f"{calc_results[0].get('q', 0)*100:.1f}%" if calc_results else "0,0%"
            },
            "tabelas": {}
        }

        # 2. Construção das Tabelas Dinâmicas (Listas de Dicts)
        
        # Tabela: Cronograma
        tbl_cronograma = []
        tbl_strikes = []
        tbl_vol = []
        tbl_taxa = []
        tbl_resultados = []
        
        data_outorga = general_info.get("data_outorga", date.today())
        
        for i, row in enumerate(calc_results):
            lote_nome = f"Lote {row.get('Tranche', i+1)}"
            vesting_years = row.get('Vesting', 0)
            life_years = row.get('T', 0)
            
            dt_vesting = data_outorga + timedelta(days=int(vesting_years*365))
            dt_venc = data_outorga + timedelta(days=int(life_years*365))
            
            # Cronograma
            tbl_cronograma.append({
                "lote": lote_nome,
                "numero": i+1,
                "qtd": "A definir", # Dado que geralmente vem de um arquivo de beneficiários
                "data_outorga": ReportService._format_date(data_outorga),
                "data_vesting": ReportService._format_date(dt_vesting),
                "data_vencimento": ReportService._format_date(dt_venc)
            })

            # Strikes (Preço de Exercício)
            tbl_strikes.append({
                "lote": lote_nome,
                "strike": ReportService._format_currency(row.get('K', 0))
            })
            
            # Volatilidade
            tbl_vol.append({
                "data_base": ReportService._format_date(data_outorga),
                "vencimento": ReportService._format_date(dt_venc),
                "valor": f"{row.get('Vol', 0)*100:.2f}%"
            })
            
            # Taxa Livre de Risco
            tbl_taxa.append({
                "lote": lote_nome,
                "vencimento": ReportService._format_date(dt_venc),
                "taxa": f"{row.get('r', 0)*100:.2f}%"
            })
            
            # Resultado Final
            tbl_resultados.append({
                "lote": lote_nome,
                "modelo": analysis_result.model_recommended.value,
                "fv_final": ReportService._format_currency(row.get('FV Unit', 0))
            })

        context["tabelas"]["cronograma"] = tbl_cronograma
        context["tabelas"]["strikes"] = tbl_strikes
        context["tabelas"]["volatilidade"] = tbl_vol
        context["tabelas"]["taxa_livre_risco"] = tbl_taxa
        context["tabelas"]["resultados_fair_value"] = tbl_resultados
        
        # Tabela de Projeção de Despesas (Simplificada - Linear)
        # Numa versão real, isso faria o "amortization schedule"
        tbl_despesas = []
        total_cost = sum([r.get('FV Ponderado', 0) for r in calc_results])
        
        # Mock de 3 anos de despesa
        for ano_idx in range(3):
            tbl_despesas.append({
                "ano": data_outorga.year + ano_idx,
                "custo_ilp": ReportService._format_currency(total_cost / 3),
                "custo_encargos": "R$ 0,00",
                "total": ReportService._format_currency(total_cost / 3)
            })
        context["tabelas"]["projecao_despesas"] = tbl_despesas

        return context

    @staticmethod
    def render_template(template_file, context) -> io.BytesIO:
        """
        Carrega o template, renderiza com o contexto e retorna bytes.
        """
        doc = DocxTemplate(template_file)
        doc.render(context)
        
        output = io.BytesIO()
        doc.save(output)
        output.seek(0)
        return output
