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
    Versão Otimizada: Integra lógica robusta de scraping da B3 (Sistema Pregão).
    """

    # =========================================================================
    # MÓDULO 1: VOLATILIDADE (Yahoo Finance + Arch) - MANTIDO
    # =========================================================================
    
    @staticmethod
    def get_peer_group_volatility(tickers: List[str], start_date: date, end_date: date) -> Dict[str, Any]:
        """Calcula volatilidade histórica (Mantido da versão anterior)."""
        results = {}
        valid_ewma_values = []
        valid_std_values = []
        valid_garch_values = []
        s_str = start_date.strftime("%Y-%m-%d")
        e_str = end_date.strftime("%Y-%m-%d")

        for ticker_raw in tickers:
            ticker = ticker_raw.strip().upper()
            if not ticker.endswith(".SA") and any(char.isdigit() for char in ticker):
                ticker += ".SA"
            try:
                data = yf.download(ticker, start=s_str, end=e_str, progress=False)
                if data.empty:
                    results[ticker_raw] = {"error": "Sem dados retornados."}
                    continue
                
                if 'Adj Close' in data.columns: series = data['Adj Close']
                elif 'Close' in data.columns: series = data['Close']
                else:
                    results[ticker_raw] = {"error": "Colunas de preço ausentes."}
                    continue

                if isinstance(series, pd.DataFrame): series = series.iloc[:, 0]
                
                returns = np.log(series / series.shift(1)).dropna()
                if len(returns) < 30:
                    results[ticker_raw] = {"error": f"Amostra insuficiente ({len(returns)} dias)."}
                    continue

                std_vol = returns.std() * np.sqrt(252)
                ewma_vol = returns.ewm(alpha=(1 - 0.94)).std().iloc[-1] * np.sqrt(252)
                
                garch_vol = None
                try:
                    r_scaled = returns * 100 
                    model = arch_model(r_scaled, vol='Garch', p=1, q=1, rescale=False)
                    res_model = model.fit(disp='off', show_warning=False)
                    forecast = res_model.forecast(horizon=1)
                    garch_vol = np.sqrt(forecast.variance.iloc[-1, 0] * 252) / 100
                except: garch_vol = None

                valid_std_values.append(std_vol)
                valid_ewma_values.append(ewma_vol)
                if garch_vol: valid_garch_values.append(garch_vol)

                results[ticker_raw] = {
                    "std_dev": std_vol, "ewma": ewma_vol, "garch": garch_vol,
                    "last_price": float(series.iloc[-1]), "samples": len(returns),
                    "last_date": returns.index[-1].strftime('%d/%m/%Y')
                }
            except Exception as e:
                results[ticker_raw] = {"error": str(e)}

        summary = {
            "mean_std": np.mean(valid_std_values) if valid_std_values else 0.0,
            "mean_ewma": np.mean(valid_ewma_values) if valid_ewma_values else 0.0,
            "mean_garch": np.mean(valid_garch_values) if valid_garch_values else 0.0,
            "count_valid": len(valid_std_values)
        }
        return {"summary": summary, "details": results}

    # =========================================================================
    # MÓDULO 2: CURVA DE JUROS DI (B3 Scraping Otimizado - CORRIGIDO)
    # =========================================================================

    @staticmethod
    def _parse_b3_maturity_code(di_code: str) -> Optional[date]:
        """Converte códigos de vencimento (Ex: F26, JAN/26) em objeto date."""
        meses = {"F": 1, "JAN": 1, "G": 2, "FEV": 2, "H": 3, "MAR": 3, 
                 "J": 4, "ABR": 4, "APR": 4, "K": 5, "MAI": 5, "MAY": 5, 
                 "M": 6, "JUN": 6, "N": 7, "JUL": 7, "Q": 8, "AGO": 8, "AUG": 8, 
                 "U": 9, "SET": 9, "SEP": 9, "V": 10, "OUT": 10, "OCT": 10, 
                 "X": 11, "NOV": 11, "Z": 12, "DEZ": 12, "DEC": 12}
        
        di_code = str(di_code).strip().upper()
        
        # Caso 1: Código curto "F26" (Padrão antigo ou ticker)
        match_code = re.match(r"([A-Z])(\d{2})", di_code)
        if match_code:
            mes_letra, ano_dois_digitos = match_code.groups()
            if mes_letra in meses:
                return date(2000 + int(ano_dois_digitos), meses[mes_letra], 1)

        # Caso 2: Formato JAN/26 ou JAN/2026 (Padrão planilha B3)
        match_slash = re.match(r"([A-Z]+)/(\d{2,4})", di_code)
        if match_slash:
            mes_str, ano_str = match_slash.groups()
            mes = meses.get(mes_str[:3]) 
            if mes:
                ano = int(ano_str)
                if ano < 100: ano += 2000
                return date(ano, mes, 1)

        return None

    @staticmethod
    def gerar_url_di(data_ref: date) -> str:
        # CORREÇÃO CRÍTICA: Força o formato DD/MM/AAAA para a URL da B3
        d_fmt = data_ref.strftime("%d/%m/%Y")
        return f"https://www2.bmf.com.br/pages/portal/bmfbovespa/boletim1/SistemaPregao_excel1.asp?Data={d_fmt}&Mercadoria=DI1&XLS=true"

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def get_di_data_b3(reference_date: date) -> pd.DataFrame:
        """
        Busca a curva DI completa usando a lógica robusta do script funcional.
        """
        url = MarketDataService.gerar_url_di(reference_date)
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0"})

        try:
            response = session.get(url, timeout=15)
            response.raise_for_status()
            
            # Leitura robusta
            dfs = pd.read_html(response.content, encoding='latin1', decimal=',', thousands='.')
            if not dfs: return pd.DataFrame()

            # Lógica de Busca da Tabela (Igual ao script funcional)
            df_alvo = None
            
            # Tenta direto a tabela 6 (comum) ou busca por conteúdo
            if len(dfs) >= 7:
                 # Verificação rápida
                 if 'VENC' in str(dfs[6].values).upper():
                     df_alvo = dfs[6]
            
            if df_alvo is None:
                for df in dfs:
                    s_df = str(df.values).upper()
                    # Procura colunas chave
                    if 'VENC' in s_df and ('AJUSTE' in s_df or 'ÚLT. PREÇO' in s_df):
                        df_alvo = df
                        break
            
            if df_alvo is None or len(df_alvo) < 2: return pd.DataFrame()

            df = df_alvo.copy()
            
            # Limpeza de cabeçalho: Procura a linha que contém "VENC" e "AJUSTE"
            header_idx = -1
            for idx, row in df.iterrows():
                row_str = row.astype(str).str.upper().values
                if any("VENC" in x for x in row_str) and any("AJUSTE" in x or "PREÇO" in x for x in row_str):
                    header_idx = idx
                    break
            
            if header_idx != -1:
                df.columns = df.iloc[header_idx]
                df = df.iloc[header_idx+1:].reset_index(drop=True)

            # Mapa de colunas do script funcional
            mapa_colunas = {
                'VENC.': 'Vencimento_Str', 
                'AJUSTE': 'Taxa',
                'PREÇO AJUSTE': 'Taxa',
                'ÚLT. PREÇO': 'Taxa', 
                'ULT. PREÇO': 'Taxa'
            }
            
            cols_found = {}
            for c in df.columns:
                c_clean = str(c).strip().upper()
                if c_clean in mapa_colunas:
                    cols_found[c] = mapa_colunas[c_clean]
            
            df = df.rename(columns=cols_found)
            
            if 'Vencimento_Str' not in df.columns or 'Taxa' not in df.columns:
                return pd.DataFrame()

            clean_data = []
            for _, row in df.iterrows():
                try:
                    venc_str = row['Vencimento_Str']
                    taxa_raw = row['Taxa']
                    
                    if pd.isna(venc_str) or pd.isna(taxa_raw): continue

                    # Converte Vencimento
                    dt_venc = MarketDataService._parse_b3_maturity_code(venc_str)
                    if not dt_venc: continue
                    
                    # Converte Taxa
                    if isinstance(taxa_raw, str):
                        taxa_raw = taxa_raw.replace('.', '').replace(',', '.')
                    taxa_val = float(taxa_raw)
                    
                    # Normalização para decimal (ex: 12.50 -> 0.1250)
                    if taxa_val > 0.50: taxa_val = taxa_val / 100.0
                    
                    dias_corridos = (dt_venc - reference_date).days
                    
                    if dias_corridos > 0:
                        clean_data.append({
                            "Vencimento_Data": dt_venc,
                            "Vencimento_Str": str(venc_str),
                            "Dias_Corridos": dias_corridos,
                            "Taxa": taxa_val
                        })
                except:
                    continue
            
            if not clean_data: return pd.DataFrame()
            
            return pd.DataFrame(clean_data).sort_values("Vencimento_Data")

        except Exception as e:
            print(f"Erro request B3: {e}")
            return pd.DataFrame()

    @staticmethod
    def get_closest_di_vertex(target_date: date, df_di: pd.DataFrame) -> Tuple[str, float, str]:
        if df_di.empty: return ("N/A", 0.1075, "Erro: Sem dados B3")
        df_calc = df_di.copy()
        df_calc['diff_days'] = df_calc['Vencimento_Data'].apply(lambda x: abs((x - target_date).days))
        closest = df_calc.loc[df_calc['diff_days'].idxmin()]
        msg = "Sucesso"
        if closest['diff_days'] > 180: msg = f"Aviso: Vértice distante ({closest['diff_days']} dias)"
        return (closest['Vencimento_Str'], closest['Taxa'], msg)

    @staticmethod
    def interpolate_di_rate(target_years: float, curve_df: pd.DataFrame) -> float:
        if curve_df.empty: return 0.1075
        target_days = target_years * 365.0
        df_sort = curve_df.sort_values('Dias_Corridos')
        x = df_sort['Dias_Corridos'].values
        y = df_sort['Taxa'].values
        if target_days <= x[0]: return y[0]
        if target_days >= x[-1]: return y[-1]
        return float(np.interp(target_days, x, y))
