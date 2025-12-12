# services/rule_extractor.py
import re
import unicodedata
import logging
from typing import List, Set, Tuple, Dict, Any

# --- 1. Definição do Dicionário (Pode ser importado de um arquivo constants.py para limpar o código) ---
DICIONARIO_UNIFICADO_HIERARQUICO = {
    "FormularioReferencia_Item_8_4": {
        "a_TermosGerais": {"aliases": ["termos e condições gerais", "objetivos do plano", "elegíveis", "principais regras"], "subtopicos": {}},
        "b_Aprovacao": {"aliases": ["data de aprovação", "órgão responsável", "assembleia geral"], "subtopicos": {}},
        "c_MaximoAcoes": {"aliases": ["número máximo de ações abrangidas", "diluição máxima"], "subtopicos": {}},
        "d_MaximoOpcoes": {"aliases": ["número máximo de opções a serem outorgadas", "limite de opções"], "subtopicos": {}},
        "e_CondicoesAquisicao": {"aliases": ["condições de aquisição de ações", "metas de desempenho", "tempo de serviço"], "subtopicos": {}},
        "f_CriteriosPreco": {"aliases": ["critérios para fixação do preço de aquisição", "preço de exercício", "preço fixo previamente estabelecido"], "subtopicos": {}},
        "g_CriteriosPrazo": {"aliases": ["critérios para fixação do prazo de aquisição", "prazo de exercício"], "subtopicos": {}},
        "h_FormaLiquidacao": {"aliases": ["forma de liquidação", "pagamento em dinheiro", "entrega física das ações", "entrega de ações"], "subtopicos": {}},
        "i_RestricoesTransferencia": {"aliases": ["restrições à transferência", "períodos de bloqueio", "lockup", "bloqueio", "período de restrição à negociação"], "subtopicos": {}},
        "j_SuspensaoExtincao": {"aliases": ["suspensão, alteração ou extinção do plano", "mudanças nas políticas"], "subtopicos": {}},
        "k_EfeitosSaida": {"aliases": ["efeitos da saída do administrador", "regras de desligamento", "aposentadoria", "demissão"], "subtopicos": {}},
    },
    "TiposDePlano": {
        "AcoesRestritas": {
            "aliases": ["Ações Restritas", "Restricted Shares", "RSU"],
            "subtopicos": {
                "PerformanceShares": {
                    "aliases": ["Performance Shares", "PSU", "Ações de Performance"],
                    "subtopicos": {}
                }
            }
        },
        "OpcoesDeCompra": {
            "aliases": ["Opções de Compra", "Stock Options", "ESOP", "SOP"],
            "subtopicos": {}
        },
        "PlanoCompraAcoes_ESPP": {
            "aliases": ["Plano de Compra de Ações", "Employee Stock Purchase Plan", "ESPP"],
            "subtopicos": {
                "Matching_Coinvestimento": {
                    "aliases": ["Matching", "Contrapartida", "Co-investimento", "Plano de Matching"],
                    "subtopicos": {}
                }
            }
        },
        "AcoesFantasmas": {
            "aliases": ["Ações Fantasmas", "Phantom Shares", "Ações Virtuais"],
            "subtopicos": {}
        },
        "OpcoesFantasmas_SAR": {
            "aliases": ["Opções Fantasmas", "Phantom Options", "SAR", "Share Appreciation Rights", "Direito à Valorização de Ações"],
            "subtopicos": {}
        },
        "BonusRetencaoDiferido": {
            "aliases": ["Bônus de Retenção", "Bônus de Permanência", "Staying Bonus", "Retention Bonus", "Deferred Bonus"],
            "subtopicos": {}
        }
    },
    "MecanicasCicloDeVida": {
        "Outorga": {"aliases": ["Outorga", "Concessão", "Grant", "Grant Date"], "subtopicos": {}},
        "Vesting": {"aliases": ["Vesting", "Período de Carência", "Aquisição de Direitos", "cronograma de vesting", "Vesting Gradual"], "subtopicos": {}},
        "VestingCliff": {"aliases": ["Cliff", "Cliff Period", "Período de Cliff", "Carência Inicial"], "subtopicos": {}},
        "VestingAcelerado": {"aliases": ["Vesting Acelerado", "Accelerated Vesting", "Cláusula de Aceleração", "antecipação do vesting"], "subtopicos": {}},
        "VestingTranche": {"aliases": ["Tranche", "Lote", "Parcela do Vesting", 'Parcela', "Aniversário"], "subtopicos": {}},
        "PrecoExercicio": {"aliases": ["Preço de Exercício", "Strike", "Strike Price"], "subtopicos": {}},
        "PrecoDesconto": {"aliases": ["Desconto de", "preço com desconto", "desconto sobre o preço"], "subtopicos": {}},
        "CicloExercicio": {"aliases": ["Exercício", "Período de Exercício", "pagamento", "liquidação", "vencimento", "expiração"], "subtopicos": {}},
        "Lockup": {"aliases": ["Lockup", "Período de Lockup", "Restrição de Venda"], "subtopicos": {}},
    },
    "GovernancaRisco": {
        "DocumentosPlano": {"aliases": ["Regulamento", "Regulamento do Plano", "Contrato de Adesão", "Termo de Outorga"], "subtopicos": {}},
        "OrgaoDeliberativo": {"aliases": ["Comitê de Remuneração", "Comitê de Pessoas", "Deliberação do Conselho", "Conselho de Administração"], "subtopicos": {}},
        "MalusClawback": {"aliases": ["Malus", "Clawback", "Cláusula de Recuperação", "Forfeiture", "SOG", "Stock Ownership Guidelines"], "subtopicos": {}},
        "Diluicao": {"aliases": ["Diluição", "Dilution", "Capital Social", "Fator de Diluição"], "subtopicos": {}},
        "NaoConcorrencia": {
            "aliases": ["Non-Compete", "Não-Competição", "Garden-leave", "obrigação de não concorrer", "proibição de competição"],
            "subtopicos": {}
        }
    },
    "ParticipantesCondicoes": {
        "Elegibilidade": {"aliases": ["Participantes", "Beneficiários", "Elegíveis", "Empregados", "Administradores", "Colaboradores", "Executivos", "Diretores", "Gerentes", "Conselheiros"], "subtopicos": {}},
        "CondicaoSaida": {"aliases": ["Desligamento", "Saída", "Término do Contrato", "Rescisão", "Demissão", "Good Leaver", "Bad Leaver"], "subtopicos": {}},
        "CasosEspeciais": {"aliases": ["Aposentadoria", "Morte", "Invalidez", "Afastamento"], "subtopicos": {}},
    },
    "IndicadoresPerformance": {
        "ConceitoGeral_Performance": {
            "aliases": ["Plano de Desempenho", "Metas de Performance", "critérios de desempenho", "metas", "indicadores de performance", "metas", "performance"],
            "subtopicos": {
                "Financeiro": {
                    "aliases": [
                        "ROIC", "EBITDA", "LAIR", "Lucro", "CAGR", "Receita Líquida",
                        "fluxo de caixa", "geração de caixa", "Free Cash Flow", "FCF",
                        "lucros por ação", "Earnings per Share", "EPS", "redução de dívida",
                        "Dívida Líquida / EBITDA", "capital de giro", "retorno sobre investimentos",
                        "retorno sobre capital", "Return on Investment", "ROCE",
                        "margem bruta", "margem operacional", "lucro líquido",
                        "lucro operacional", "receita operacional", "vendas líquidas",
                        "valor econômico agregado", "custo de capital", "WACC",
                        "Weighted Average Capital Cost", "retorno sobre ativo",
                        "retorno sobre ativo líquido", "rotatividade de ativos líquidos",
                        "rotatividade do estoque", "despesas de capital", "dívida financeira bruta",
                        "receita operacional líquida", "lucros por ação diluídos",
                        "lucros por ação básicos", "rentabilidade", "Enterprise Value", "EV",
                        "Valor Teórico da Companhia", "Valor Teórico Unitário da Ação",
                        "Economic Value Added", "EVA", "NOPAT", "Net Operating Profit After Tax",
                        "Capital Total Investido", "CAGR EBITDA per Share", "Equity Value"
                    ],
                    "subtopicos": {}
                },
                "Mercado": {
                    "aliases": ["CDI", "IPCA", "Selic"],
                    "subtopicos": {}
                },
                "TSR": {
                    "aliases": ["TSR", "Total Shareholder Return", "Retorno Total ao Acionista"],
                    "subtopicos": {
                        "TSR_Absoluto": {"aliases": ["TSR Absoluto"], "subtopicos": {}},
                        "TSR_Relativo": {"aliases": ["TSR Relativo", "Relative TSR", "TSR versus", "TSR comparado a"], "subtopicos": {}}
                    }
                },
                "ESG": {
                    "aliases": ["Metas ESG", "ESG", "Neutralização de Emissões", "Redução de Emissões", "Igualdade de Gênero", "objetivos de desenvolvimento sustentável", "IAGEE", "ICMA"],
                    "subtopicos": {}
                },
                "Operacional": {
                    "aliases": ["produtividade", "eficiência operacional", "desempenho de entrega", "desempenho de segurança", "qualidade", "satisfação do cliente", "NPS", "conclusão de aquisições", "expansão comercial", "crescimento"],
                    "subtopicos": {}
                }
            }
        },
        "GrupoDeComparacao": {"aliases": ["Peer Group", "Empresas Comparáveis", "Companhias Comparáveis"], "subtopicos": {}}
    },
    "EventosFinanceiros": {
        "EventosCorporativos": {"aliases": ["grupamento", "desdobramento", "cisão", "fusão", "incorporação", "bonificação"], "subtopicos": {}},
        "MudancaDeControle": {"aliases": ["Mudança de Controle", "Change of Control", "Transferência de Controle"], "subtopicos": {}},
        "DividendosProventos": {"aliases": ["Dividendos", "JCP", "Juros sobre capital próprio", "dividend equivalent", "proventos"], "subtopicos": {}},
        "EventosDeLiquidez": {
            "aliases": ["Evento de Liquidez", "liquidação antecipada", "saída da companhia", "transação de controle", "reorganização societária", "desinvestimento", "deslistagem", "Operação Relevante"],
            "subtopicos": {
                "IPO_OPI": {"aliases": ["IPO", "Oferta Pública Inicial", "Oferta Publica Inicial", "abertura do capital"], "subtopicos": {}},
                "AlienacaoControle": {"aliases": ["Alienação de Controle", "alienação de mais de 50% de ações ordinárias", "venda de controle", "transferência de controle acionário", "venda ou permuta de Ações"], "subtopicos": {}},
                "FusaoAquisicaoVenda": {"aliases": ["Fusão", "Aquisição", "Incorporação", "Venda da Companhia", "venda, locação, arrendamento, cessão, licenciamento, transferência ou qualquer outra forma de disposição da totalidade ou de parte substancial dos ativos"], "subtopicos": {}},
                "InvestimentoRelevante": {"aliases": ["investimento primário de terceiros", "aumento de capital", "Capitalização da Companhia"], "subtopicos": {}}
            }
        }
    },
    "AspectosFiscaisContabeis": {
        "TributacaoEncargos": {"aliases": ["Encargos", "Impostos", "Tributação", "Natureza Mercantil", "Natureza Remuneratória", "INSS", "IRRF"], "subtopicos": {}},
        "NormasContabeis": {"aliases": ["IFRS 2", "CPC 10", "Valor Justo", "Fair Value", "Black-Scholes", "Despesa Contábil", "Volatilidade"], "subtopicos": {}},
    }
}

class RuleBasedExtractor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Compila as Regex apenas uma vez na inicialização da classe
        self.regex_fatos = self._compile_fact_regexes()
        self.dicionario = DICIONARIO_UNIFICADO_HIERARQUICO

    def _compile_fact_regexes(self) -> Dict[str, re.Pattern]:
        """Pré-compila regex para performance."""
        num_pattern = r'(\d{1,3}(?:[.,]\d{3})*|\d+|um|uma|dois|duas|tres|três|quatro|cinco|seis|sete|oito|nove|dez)'
        unit_pattern_anomesdias = r'\b(anos?|meses|dias)\b'

        return {
            'vesting': re.compile(
                fr'(?:vesting|periodo\s+de\s+carencia|prazo\s+de\s+carencia|periodo\s+de\s+aquisicao)'
                fr'[\s\S]{{0,250}}?(?:de\s+)?{num_pattern}\s*\(?[\s\S]{{0,100}}?{unit_pattern_anomesdias}', re.IGNORECASE
            ),
            'lockup': re.compile(
                fr'(?:lock-up|periodo\s+de\s+restricao|restricao\s+a\s+venda)'
                fr'[\s\S]{{0,250}}?(?:de\s+)?{num_pattern}\s*\(?[\s\S]{{0,100}}?{unit_pattern_anomesdias}', re.IGNORECASE
            ),
            'diluicao': re.compile(
                r'(?:diluicao|limite|nao\s+exceda|representativas\s+de|no\s+maximo)'
                r'[\s\S]{0,250}?\b(\d{1,3}(?:[.,]\d{1,2})?)\s*%', re.IGNORECASE
            ),
            'desconto': re.compile(
                r'(?:desconto|desagio|abatimento|reducao)[\s\S]{0,100}?\b(\d{1,3}(?:[.,]\d{1,2})?)\s*%', re.IGNORECASE
            ),
            'malus_clawback': re.compile(r'\b(malus|clawback|clausula\s+de\s+recuperacao|forfeiture)\b', re.IGNORECASE),
            'dividendos': re.compile(r'(?:dividendos|jcp|juros\s+sobre\s+capital\s+proprio)[\s\S]{0,200}?(?:durante|no\s+periodo\s+de)\s*(?:carencia|vesting)', re.IGNORECASE),
            'tsr': re.compile(r'\b(retorno\s+total\s+d[oa]s\s+acionistas|total\s+shareholder\s+return|tsr)\b', re.IGNORECASE),
            'roic': re.compile(r'\b(retorno\s+sobre\s+o\s+capital\s+investido|return\s+on\s+invested\s+capital|roic)\b', re.IGNORECASE),
        }

    def normalizar_texto(self, texto: str) -> str:
        if not isinstance(texto, str): return ""
        texto = texto.lower()
        return ''.join(c for c in unicodedata.normalize('NFD', texto) if unicodedata.category(c) != 'Mn')

    def _converter_palavra_para_int(self, palavra: str) -> int:
        # (Sua lógica original de conversão)
        if not isinstance(palavra, str): return 0
        palavra_limpa = self.normalizar_texto(palavra).strip()
        mapeamento = {'um': 1, 'uma': 1, 'dois': 2, 'duas': 2, 'tres': 3, 'quatro': 4, 'cinco': 5, 'seis': 6, 'sete': 7, 'oito': 8, 'nove': 9, 'dez': 10}
        if palavra_limpa in mapeamento: return mapeamento[palavra_limpa]
        try:
            return float(palavra_limpa.replace(',', '.'))
        except (ValueError, TypeError):
            return 0

    def _recursive_topic_finder(self, texto_limpo: str, sub_dict: Dict, path_so_far: List[str], found_items: Set):
        """Versão adaptada para método de instância."""
        for topic_key, topic_data in sub_dict.items():
            current_path = path_so_far + [topic_key]
            sorted_aliases = sorted(topic_data.get("aliases", []), key=len, reverse=True)
            
            for alias in sorted_aliases:
                alias_norm = self.normalizar_texto(alias)
                # Regex boundary \b é crucial aqui
                if re.search(r'\b' + re.escape(alias_norm) + r'\b', texto_limpo):
                    found_items.add((tuple(current_path), alias))
                    break # Encontrou um alias para este tópico, não precisa buscar os outros aliases do mesmo tópico

            if "subtopicos" in topic_data and topic_data["subtopicos"]:
                self._recursive_topic_finder(texto_limpo, topic_data["subtopicos"], current_path, found_items)

    def extract_facts(self, text: str) -> Dict[str, Any]:
        """Extrai fatos quantitativos e booleanos (TSR, Vesting, etc)."""
        facts = {}
        text_norm = self.normalizar_texto(text)
        text_clean = re.sub(r'[^\w\s]', ' ', text_norm) # Sem pontuação para booleanos

        # --- Lógica de Extração (Vesting) ---
        if match := self.regex_fatos['vesting'].search(text_norm):
            val = self._converter_palavra_para_int(match.group(1))
            unit = match.group(2).lower().rstrip('s')
            # Normaliza para anos
            years = val if unit == 'ano' else val / 12.0 if unit == 'mes' else val / 365.0
            if 0 < years < 20: 
                facts['vesting_period'] = round(years, 2)

        # --- Lógica de Extração (Diluição) ---
        if match := self.regex_fatos['diluicao'].search(text_norm):
            val = float(match.group(1).replace(',', '.'))
            if 0 < val < 50: # Filtro de sanidade
                facts['dilution_cap'] = val

        # --- Booleanos ---
        facts['has_malus_clawback'] = bool(self.regex_fatos['malus_clawback'].search(text_clean))
        facts['has_tsr'] = bool(self.regex_fatos['tsr'].search(text_clean))
        facts['has_roic'] = bool(self.regex_fatos['roic'].search(text_clean))
        
        return facts

    def analyze_single_plan(self, text: str) -> Dict[str, Any]:
        """
        Método principal para ser chamado pelo Icarus.
        Recebe o texto COMPLETO de um único plano.
        """
        text_norm = self.normalizar_texto(text)
        text_clean = re.sub(r'[^\w\s]', ' ', text_norm)
        
        # 1. Identificar Tópicos e Tipos de Plano
        found_topics = set()
        self._recursive_topic_finder(text_clean, self.dicionario, [], found_topics)
        
        # Estruturar tópicos encontrados
        topics_structure = {}
        plan_types_found = []
        
        for path_tuple, alias in found_topics:
            path_list = list(path_tuple)
            # Se for do ramo "TiposDePlano", adiciona à lista principal
            if path_list[0] == "TiposDePlano":
                plan_types_found.append(path_list[-1]) # Pega o nó folha (ex: "StockOptions")
            
            # Adiciona à estrutura hierárquica (simplificada para visualização)
            key = "/".join(path_list)
            topics_structure[key] = alias

        # 2. Extrair Fatos Específicos (Regex)
        facts = self.extract_facts(text)

        # 3. Retornar DTO (Data Transfer Object) consolidado
        return {
            "detected_plan_types": list(set(plan_types_found)),
            "extracted_facts": facts,
            "topic_matches": topics_structure
        }
