import pandas as pd
import numpy as np
import yfinance as yf
from arch import arch_model
from datetime import date, datetime
import requests
import re
from typing import List, Dict, Any, Tuple
import streamlit as st

class MarketDataService:
    """
    Serviço de Dados de Mercado (Backend).
    Versão Final: Integração total com a lógica do Código de Referência (Tabela 6 + Formatação).
    """

    # =========================================================================
    # MÓDULO 1: VOLATILIDADE (Yahoo Finance + Arch) - MANTIDO
    # =========================================================================
    @staticmethod
    def get_peer_group_volatility(tickers: List[str], start_date: date, end_date: date) -> Dict[str, Any]:
        results = {}
        valid_std_values = []
        valid_ewma_values = []
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
                    results[ticker_raw] = {"error": "Sem dados"}
                    continue
                
                # Seleção de coluna robusta
                series = None
                if 'Adj Close' in data.columns: series = data['Adj Close']
                elif 'Close' in data.columns: series = data['Close']
                
                if series is None:
                    results[ticker_raw] = {"error": "Sem coluna de preço"}
                    continue
                    
                if isinstance(series, pd.DataFrame): series = series.iloc[:, 0]
                
                returns = np.log(series / series.shift(1)).dropna()
                if len(returns) < 30:
                    results[ticker_raw] = {"error": "Amostra insuficiente"}
                    continue

                # Cálculos
                std_vol = returns.std() * np.sqrt(252)
                ewma_vol = returns.ewm(alpha=(1 - 0.94)).std().iloc[-1] * np.sqrt(252)
                
                garch_vol = None
                try:
                    r_scaled = returns * 100 
                    model = arch_model(r_scaled, vol='Garch', p=1, q=1, rescale=False)
                    res = model.fit(disp='off', show_warning=False)
                    garch_vol = np.sqrt(res.forecast(horizon=1).variance.iloc[-1, 0] * 252) / 100
                except: pass

                valid_std_values.append(std_vol)
                valid_ewma_values.append(ewma_vol)
                if garch_vol: valid_garch_values.append(garch_vol)

                results[ticker_raw] = {
                    "std_dev": std_vol, "ewma": ewma_vol, "garch": garch_vol,
                    "last_price": float(series.iloc[-1])
                }
            except Exception as e:
                results[ticker_raw] = {"error": str(e)}

        return {
            "summary": {
                "mean_std": np.mean(valid_std_values) if valid_std_values else 0.0,
                "mean_ewma": np.mean(valid_ewma_values) if valid_ewma_values else 0.0,
                "count_valid": len(valid_std_values)
            },
            "details": results
        }

    # =========================================================================
    # MÓDULO 2: CURVA DE JUROS DI (Lógica do Código de Referência)
    # =========================================================================

    @staticmethod
    def gerar_url_di(data_ref: date) -> str:
        # Garante DD/MM/AAAA
        if isinstance(data_ref, str):
            try: data_ref = datetime.strptime(data_ref, "%Y-%m-%d").date()
            except: pass
        d_fmt = data_ref.strftime("%d/%m/%Y")
        return f"https://www2.bmf.com.br/pages/portal/bmfbovespa/boletim1/SistemaPregao_excel1.asp?Data={d_fmt}&Mercadoria=DI1&XLS=true"

    @staticmethod
    def converter_vencimento_ref(di_code):
        """Lógica exata do código de referência para converter F26 -> 01/2026"""
        meses = {"F": "01", "G": "02", "H": "03", "J": "04", "K": "05", "M": "06", 
                 "N": "07", "Q": "08", "U": "09", "V": "10", "X": "11", "Z": "12"}
        if isinstance(di_code, str) and len(di_code) == 3 and di_code[0] in meses:
            try:
                ano = 2000 + int(di_code[1:])
                return f"{meses[di_code[0]]}/{ano}"
            except: return ""
        return ""

    @staticmethod
    @st.cache_data(ttl=3600, show_spinner=False)
    def get_di_data_b3(reference_date: date) -> pd.DataFrame:
        url = MarketDataService.gerar_url_di(reference_date)
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0"})

        try:
            response = session.get(url, timeout=15)
            # 1. Leitura com parâmetros do código de referência
            tabelas_dfs = pd.read_html(response.content, encoding='latin1', decimal=',', thousands='.')

            if len(tabelas_dfs) < 7: return pd.DataFrame() # Tabela 7 ausente

            # 2. Seleção direta da Tabela 6
            df = tabelas_dfs[6]
            if len(df) < 2: return pd.DataFrame()

            # 3. Limpeza de Cabeçalho (Linha 1 é header, dados começam na 2)
            df.columns = df.iloc[1]
            df = df.iloc[2:].reset_index(drop=True)

            # Remove linha de totais se houver
            if df.iloc[-1, 0] is None or pd.isna(df.iloc[-1, 0]):
                df = df.iloc[:-1]

            # 4. Mapeamento de Colunas (Referência)
            mapa_colunas = {
                'VENC.': 'VENCIMENTO', 
                'ÚLT. PREÇO': 'ULTIMO PRECO',
                'ULT. PREÇO': 'ULTIMO PRECO', # Fallback de nome
                'AJUSTE': 'PRECO AJUSTE'      # Apenas para garantir
            }
            # Renomeia colunas existentes
            cols_found = {}
            for c in df.columns:
                c_clean = str(c).strip().upper()
                if c_clean in mapa_colunas:
                    cols_found[c] = mapa_colunas[c_clean]
            
            df = df.rename(columns=cols_found)

            if 'VENCIMENTO' not in df.columns or 'ULTIMO PRECO' not in df.columns:
                return pd.DataFrame()

            # 5. Processamento dos Dados para o Icarus
            clean_data = []
            
            for _, row in df.iterrows():
                try:
                    venc_cod = row['VENCIMENTO']
                    taxa_raw = row['ULTIMO PRECO']
                    
                    if pd.isna(venc_cod) or pd.isna(taxa_raw): continue
                    
                    # Formato Visual (01/2026) usando função de referência
                    venc_fmt = MarketDataService.converter_vencimento_ref(venc_cod)
                    if not venc_fmt: continue
                    
                    # Data Objeto (necessário para dias corridos)
                    dt_venc = datetime.strptime(venc_fmt, "%m/%Y").date()
                    
                    # Conversão Numérica (pd.read_html já deve ter tratado vírgula se decimal=',')
                    # Mas garantimos conversão segura
                    taxa_val = float(taxa_raw)
                    
                    # 6. Escala (Normalização para Decimal)
                    # Se B3 retorna 14.89 (float), Icarus precisa de 0.1489
                    # Se read_html falhou e leu 1489.0, ajustamos também
                    if taxa_val > 100:
                         taxa_val = taxa_val / 1000.0 # Caso extremo
                    elif taxa_val > 50:
                         taxa_val = taxa_val / 100.0
                    elif taxa_val > 0.50:
                         taxa_val = taxa_val / 100.0
                    
                    dias_corridos = (dt_venc - reference_date).days
                    
                    if dias_corridos > 0:
                        clean_data.append({
                            "Vencimento_Fmt": venc_fmt,     # Visual (01/2026)
                            "Vencimento_Str": str(venc_cod),# Código (F26)
                            "Vencimento_Data": dt_venc,     # Objeto Date
                            "Dias_Corridos": dias_corridos,
                            "Taxa": taxa_val
                        })
                except: continue

            return pd.DataFrame(clean_data).sort_values("Dias_Corridos")

        except Exception as e:
            print(f"Erro B3: {e}")
            return pd.DataFrame()

    @staticmethod
    def get_closest_di_vertex(target_date: date, df_di: pd.DataFrame) -> Tuple[str, float, str]:
        if df_di.empty: return ("N/A", 0.1075, "Erro: Sem dados B3")
        df_calc = df_di.copy()
        df_calc['diff_days'] = df_calc['Vencimento_Data'].apply(lambda x: abs((x - target_date).days))
        closest = df_calc.loc[df_calc['diff_days'].idxmin()]
        # Retorna formatado para UI
        return (closest['Vencimento_Fmt'], closest['Taxa'], "Sucesso")

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
