import pandas as pd
import numpy as np
import yfinance as yf
from arch import arch_model
from datetime import date, datetime, timedelta
import requests
import re
from typing import List, Dict, Optional, Union, Any, Tuple
import streamlit as st

class MarketDataService:
    """
    Serviço de Dados de Mercado (Backend).
    Responsável por:
    1. Buscar e calcular volatilidade histórica de Peer Groups (Yahoo Finance).
    2. Buscar e estruturar a Curva de Juros DI1 (B3) via HTTP Request.
    3. Encontrar o vértice de juros mais próximo para uma data alvo.
    """

    # =========================================================================
    # MÓDULO 1: VOLATILIDADE (Yahoo Finance + Arch)
    # =========================================================================
    
    @staticmethod
    def get_peer_group_volatility(
        tickers: List[str], 
        start_date: date, 
        end_date: date
    ) -> Dict[str, Any]:
        """
        Calcula volatilidade histórica para uma lista de tickers.
        Retorna dicionário com métricas individuais e a média EWMA do grupo.
        
        Métricas calculadas:
        - Std Dev (Desvio Padrão Anualizado)
        - EWMA (Exponentially Weighted Moving Average, lambda=0.94)
        - GARCH(1,1) (Generalized Autoregressive Conditional Heteroskedasticity)
        """
        results = {}
        valid_ewma_values = []
        valid_std_values = []
        valid_garch_values = []

        # Garante formato de string para yfinance
        s_str = start_date.strftime("%Y-%m-%d")
        e_str = end_date.strftime("%Y-%m-%d")

        for ticker_raw in tickers:
            ticker = ticker_raw.strip().upper()
            
            # Tratamento de sufixo .SA para ativos brasileiros
            # Evita adicionar em ativos globais (se não tiver números, assume global, ex: AAPL)
            if not ticker.endswith(".SA") and any(char.isdigit() for char in ticker):
                ticker += ".SA"
            
            try:
                # Download de dados
                data = yf.download(ticker, start=s_str, end=e_str, progress=False)
                
                if data.empty:
                    results[ticker_raw] = {"error": "Sem dados retornados pelo Yahoo Finance."}
                    continue

                # Tratamento para MultiIndex (mudança recente no yfinance)
                # Tenta pegar 'Adj Close', se não, pega 'Close'
                if 'Adj Close' in data.columns:
                    series = data['Adj Close']
                elif 'Close' in data.columns:
                    series = data['Close']
                else:
                    results[ticker_raw] = {"error": "Colunas de preço não encontradas."}
                    continue

                # Se ainda for DataFrame (devido a MultiIndex), pega a primeira coluna
                if isinstance(series, pd.DataFrame):
                    series = series.iloc[:, 0]
                
                # Cálculo dos Retornos Logarítmicos
                # log(Pt / Pt-1)
                returns = np.log(series / series.shift(1)).dropna()
                
                # Validação de Amostra Mínima
                if len(returns) < 30:
                    results[ticker_raw] = {"error": f"Amostra insuficiente ({len(returns)} dias). Mínimo 30."}
                    continue

                # 1. Desvio Padrão Simples (Anualizado)
                std_vol = returns.std() * np.sqrt(252)
                
                # 2. EWMA (RiskMetrics Lambda = 0.94)
                # Span aproximado para lambda 0.94 é ~32 dias, mas usamos alpha explícito
                ewma_vol = returns.ewm(alpha=(1 - 0.94)).std().iloc[-1] * np.sqrt(252)
                
                # 3. GARCH(1,1)
                garch_vol = None
                try:
                    # Multiplica por 100 para estabilidade numérica do otimizador
                    r_scaled = returns * 100 
                    model = arch_model(r_scaled, vol='Garch', p=1, q=1, rescale=False)
                    res_model = model.fit(disp='off', show_warning=False)
                    forecast = res_model.forecast(horizon=1)
                    var_forecast = forecast.variance.iloc[-1, 0]
                    # Desescala (divide por 100) e anualiza (sqrt(252))
                    garch_vol = np.sqrt(var_forecast * 252) / 100
                except Exception:
                    garch_vol = None # Falha na convergência ou biblioteca

                # Armazena valores válidos para média
                valid_std_values.append(std_vol)
                valid_ewma_values.append(ewma_vol)
                if garch_vol: valid_garch_values.append(garch_vol)

                results[ticker_raw] = {
                    "std_dev": std_vol,
                    "ewma": ewma_vol,
                    "garch": garch_vol,
                    "last_price": float(series.iloc[-1]),
                    "samples": len(returns),
                    "last_date": returns.index[-1].strftime('%d/%m/%Y')
                }

            except Exception as e:
                results[ticker_raw] = {"error": str(e)}

        # Resumo do Grupo
        summary = {
            "mean_std": np.mean(valid_std_values) if valid_std_values else 0.0,
            "mean_ewma": np.mean(valid_ewma_values) if valid_ewma_values else 0.0,
            "mean_garch": np.mean(valid_garch_values) if valid_garch_values else 0.0,
            "count_valid": len(valid_std_values)
        }

        return {"summary": summary, "details": results}

    # =========================================================================
    # MÓDULO 2: CURVA DE JUROS DI (B3 Scraping)
    # =========================================================================

    @staticmethod
    def _parse_b3_maturity(venc_str: str) -> Optional[date]:
        """
        Converte códigos de vencimento da B3 em objetos date.
        Formatos aceitos: '01/01/2026', 'JAN/26', 'F26'
        """
        venc_str = str(venc_str).strip().upper()
        
        # Caso 1: Formato data completa DD/MM/YYYY
        try:
            return datetime.strptime(venc_str, "%d/%m/%Y").date()
        except ValueError:
            pass

        # Mapa de Meses B3
        mapa_meses = {
            "F": 1, "JAN": 1,
            "G": 2, "FEV": 2,
            "H": 3, "MAR": 3,
            "J": 4, "ABR": 4, "APR": 4,
            "K": 5, "MAI": 5, "MAY": 5,
            "M": 6, "JUN": 6,
            "N": 7, "JUL": 7,
            "Q": 8, "AGO": 8, "AUG": 8,
            "U": 9, "SET": 9, "SEP": 9,
            "V": 10, "OUT": 10, "OCT": 10,
            "X": 11, "NOV": 11,
            "Z": 12, "DEZ": 12, "DEC": 12
        }

        # Caso 2: Códigos Curtos (ex: F26) ou Mes/Ano (JAN/26)
        # Regex para separar Letras de Números
        match = re.match(r"([A-Z]+)/?(\d{2,4})", venc_str)
        if match:
            mes_code = match.group(1)
            ano_code = match.group(2)
            
            mes = mapa_meses.get(mes_code)
            if mes:
                ano = int(ano_code)
                # Ajuste ano de 2 dígitos
                if ano < 100:
                    ano += 2000
                # Vencimento DI é sempre no 1º dia útil do mês
                return date(ano, mes, 1)
        
        return None

    @staticmethod
    def gerar_url_di(data_ref: date) -> str:
        """Gera URL direta para o Excel/HTML da B3."""
        d_fmt = data_ref.strftime("%d/%m/%Y")
        # Mercadoria DI1, XLS=true força retorno tabular limpo
        return f"https://www2.bmf.com.br/pages/portal/bmfbovespa/boletim1/SistemaPregao_excel1.asp?Data={d_fmt}&Mercadoria=DI1&XLS=true"

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def get_di_data_b3(reference_date: date) -> pd.DataFrame:
        """
        Busca a curva DI completa para uma data de referência.
        Utiliza requests + pd.read_html para robustez (sem Selenium).
        """
        url = MarketDataService.gerar_url_di(reference_date)
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        })

        try:
            response = session.get(url, timeout=15)
            response.raise_for_status()
            
            # Tenta ler tabelas HTML
            # decimal=',' e thousands='.' são cruciais para o formato brasileiro
            dfs = pd.read_html(response.content, encoding='latin1', decimal=',', thousands='.')
            
            if not dfs:
                return pd.DataFrame() # Nenhuma tabela

            # --- Lógica de Busca Dinâmica da Tabela ---
            df_alvo = None
            for df in dfs:
                # Converte para string para buscar palavras-chave
                # A tabela de DI sempre tem 'VENC' e ('AJUSTE' ou 'ÚLT. PREÇO')
                s_df = str(df.values).upper()
                if 'VENC' in s_df and ('AJUSTE' in s_df or 'ULT. PREÇO' in s_df or 'ÚLT. PREÇO' in s_df):
                    df_alvo = df
                    break
            
            if df_alvo is None:
                return pd.DataFrame() # Tabela não encontrada (Feriado?)

            # --- Limpeza do DataFrame ---
            df = df_alvo.copy()
            
            # Localiza a linha de cabeçalho real
            idx_header = -1
            for i, row in df.iterrows():
                row_str = str(row.values).upper()
                if "VENC" in row_str:
                    idx_header = i
                    break
            
            if idx_header != -1:
                df.columns = df.iloc[idx_header]
                df = df.iloc[idx_header+1:]
            
            # Remove linhas vazias
            df = df.dropna(how='all')
            
            # Normalização de Nomes de Colunas
            cols_map = {}
            for c in df.columns:
                c_clean = str(c).strip().upper()
                if c_clean.startswith("VENC"):
                    cols_map[c] = "Vencimento_Str"
                elif "AJUSTE" in c_clean or "ÚLT. PREÇO" in c_clean or "ULT. PREÇO" in c_clean:
                    # Preferência pelo Ajuste, mas aceita Último Preço
                    if "AJUSTE" in c_clean:
                        cols_map[c] = "Taxa_Ajuste"
                    else:
                        cols_map[c] = "Taxa_Ultimo"
            
            df = df.rename(columns=cols_map)
            
            # Garante que temos Vencimento e alguma Taxa
            if "Vencimento_Str" not in df.columns:
                return pd.DataFrame()
                
            col_taxa = "Taxa_Ajuste" if "Taxa_Ajuste" in df.columns else "Taxa_Ultimo"
            if col_taxa not in df.columns:
                return pd.DataFrame()

            # Processamento Final dos Dados
            clean_data = []
            
            for _, row in df.iterrows():
                try:
                    venc_str = row["Vencimento_Str"]
                    taxa_raw = row[col_taxa]
                    
                    # Converte Vencimento para Date Object
                    dt_venc = MarketDataService._parse_b3_maturity(venc_str)
                    if not dt_venc:
                        continue
                        
                    # Converte Taxa (ex: 12.55 -> 0.1255)
                    # O pandas read_html com decimal=',' ajuda, mas as vezes vem sujeira
                    if isinstance(taxa_raw, str):
                        taxa_raw = taxa_raw.replace('.', '').replace(',', '.')
                    
                    taxa_val = float(taxa_raw) / 100.0
                    
                    # Calcula Dias Úteis (Aprox) ou Dias Corridos para referência
                    dias_corridos = (dt_venc - reference_date).days
                    
                    if dias_corridos > 0:
                        clean_data.append({
                            "Vencimento_Data": dt_venc,
                            "Vencimento_Str": str(venc_str),
                            "Dias_Corridos": dias_corridos,
                            "Taxa": taxa_val
                        })
                except Exception:
                    continue
            
            # Retorna ordenado por vencimento
            df_final = pd.DataFrame(clean_data).sort_values("Vencimento_Data")
            return df_final

        except Exception as e:
            # Log silencioso para não quebrar a UI, retorna vazio
            print(f"Erro request B3: {e}")
            return pd.DataFrame()

    @staticmethod
    def get_closest_di_vertex(
        target_date: date, 
        df_di: pd.DataFrame
    ) -> Tuple[str, float, str]:
        """
        Encontra o vértice da curva DI mais próximo da data alvo.
        Retorna: (Vencimento_Formatado, Taxa, Mensagem_Status)
        """
        if df_di.empty:
            return ("N/A", 0.1075, "Erro: Sem dados B3")

        # Calcula a diferença absoluta em dias para encontrar o mais próximo
        # Copia para não alterar o original (cacheado)
        df_calc = df_di.copy()
        df_calc['diff_days'] = df_calc['Vencimento_Data'].apply(
            lambda x: abs((x - target_date).days)
        )
        
        # Pega o menor
        closest_row = df_calc.loc[df_calc['diff_days'].idxmin()]
        
        venc_str = closest_row['Vencimento_Str']
        taxa = closest_row['Taxa']
        diff = closest_row['diff_days']
        
        msg = "Sucesso"
        if diff > 180: # Se o mais próximo estiver a mais de 6 meses
            msg = f"Aviso: Vértice distante ({diff} dias)"
            
        return (venc_str, taxa, msg)
