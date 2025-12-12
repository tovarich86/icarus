"""
Serviço de Extração Baseada em Regras (Regex Engine).
Refatorado para ler configurações de um arquivo JSON externo.
"""

import re
import json
import unicodedata
import logging
from pathlib import Path
from typing import List, Set, Tuple, Dict, Any

class RuleBasedExtractor:
    def __init__(self, config_path: str = "config/rules_dictionary.json"):
        """
        Inicializa o extrator carregando as regras do JSON.
        
        Args:
            config_path: Caminho relativo para o arquivo de regras JSON.
        """
        self.logger = logging.getLogger(__name__)
        self.regex_fatos = self._compile_fact_regexes()
        
        # Carregamento Dinâmico do Dicionário
        self.dicionario = self._load_rules(config_path)

    def _load_rules(self, relative_path: str) -> Dict[str, Any]:
        """Carrega o dicionário de regras de forma segura."""
        try:
            # Tenta resolver o caminho relativo à raiz do projeto ou ao arquivo atual
            # Assumindo estrutura: /app/services/rule_extractor.py e /app/config/rules.json
            base_path = Path(__file__).resolve().parent.parent 
            full_path = base_path / relative_path
            
            if not full_path.exists():
                # Fallback: tenta caminho absoluto ou relativo direto se rodando da raiz
                full_path = Path(relative_path)
            
            if not full_path.exists():
                self.logger.error(f"Arquivo de regras não encontrado: {full_path}")
                return {}
                
            with open(full_path, 'r', encoding='utf-8') as f:
                return json.load(f)
                
        except Exception as e:
            self.logger.error(f"Erro ao carregar dicionário de regras: {str(e)}")
            return {}

    def _compile_fact_regexes(self) -> Dict[str, re.Pattern]:
        """Pré-compila regex para performance (Mantido da versão original)."""
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
        if not isinstance(palavra, str): return 0
        palavra_limpa = self.normalizar_texto(palavra).strip()
        mapeamento = {'um': 1, 'uma': 1, 'dois': 2, 'duas': 2, 'tres': 3, 'quatro': 4, 'cinco': 5, 'seis': 6, 'sete': 7, 'oito': 8, 'nove': 9, 'dez': 10}
        if palavra_limpa in mapeamento: return mapeamento[palavra_limpa]
        try:
            return float(palavra_limpa.replace(',', '.'))
        except (ValueError, TypeError):
            return 0

    def _recursive_topic_finder(self, texto_limpo: str, sub_dict: Dict, path_so_far: List[str], found_items: Set):
        """Busca recursiva de tópicos no dicionário."""
        if not sub_dict: return

        for topic_key, topic_data in sub_dict.items():
            current_path = path_so_far + [topic_key]
            # Ordena aliases por tamanho (maior para menor) para evitar match parcial precoce
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
        if not self.dicionario:
            self.logger.warning("Dicionário de regras vazio. Verifique o arquivo JSON.")
        
        text_norm = self.normalizar_texto(text)
        text_clean = re.sub(r'[^\w\s]', ' ', text_norm)
        
        # 1. Identificar Tópicos e Tipos de Plano
        found_topics = set()
        if self.dicionario:
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
