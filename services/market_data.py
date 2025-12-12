import pandas as pd
import numpy as np
import yfinance as yf
from arch import arch_model
from datetime import date, datetime
import requests
import io # Necessário para o arquivo em memória
from typing import List, Dict, Any, Tuple
import streamlit as st

class MarketDataService:
    """
    Serviço de Dados de Mercado (Backend).
    Versão com Auditoria Excel (Séries Temporais).
    """

    @staticmethod
    def _formatar_ptbr(val):
        """Helper para formatar float com vírgula (Padrão Excel PT-BR)."""
        if pd.isna(val): return ""
        if isinstance(val, (int, float)): return f"{val:.6f}".replace('.', ',')
        return val

    @staticmethod
    def _generate_excel_audit(details_dict: Dict, df_prices: pd.DataFrame, df_returns: pd.DataFrame, 
                              df_ewma: pd.DataFrame, df_garch: pd.DataFrame) -> bytes:
        """Gera o arquivo Excel binário com as abas de auditoria."""
        output = io.BytesIO()
        
        # 1. Montar DataFrame de Resumo a partir dos detalhes
        rows = []
        for ticker, metrics in details_dict.items():
            if "error" in metrics: continue
            rows.append({
                "Ticker": ticker,
                "Preço Ref": metrics.get("last_price"),
                "Vol Histórica (Std)": metrics.get("std_dev"),
                "Vol EWMA (0.94)": metrics.get("ewma"),
                "Vol GARCH(1,1)": metrics.get("garch")
            })
        df_resumo = pd.DataFrame(rows)
        
        # Formatação PT-BR no Resumo
        df_resumo_fmt = df_resumo.copy()
        cols_to_fmt = ["Preço Ref", "Vol Histórica (Std)", "Vol EWMA (0.94)", "Vol GARCH(1,1)"]
        for c in cols_to_fmt:
            if c in df_resumo_fmt.columns:
                df_resumo_fmt[c] = df_resumo_fmt[c].apply(MarketDataService._formatar_ptbr)

        # 2. Gravação do Excel (Engine xlsxwriter é padrão no Streamlit Cloud/Linux)
        # Usamos xlsxwriter para evitar dependência estrita de openpyxl se não houver
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df_resumo_fmt.to_excel(writer, sheet_name='Resumo', index=False)
            df_prices.to_excel(writer, sheet_name='Audit_Precos')
            df_returns.to_excel(writer, sheet_name='Audit_Retornos')
            df_ewma.to_excel(writer, sheet_name='Audit_EWMA_Serie')
            df_garch.to_excel(writer, sheet_name='Audit_GARCH_Serie')
            
        return output.getvalue()

    @staticmethod
    def get_peer_group_volatility(tickers: List[str], start_date: date, end_date: date) -> Dict[str, Any]:
        results = {}
        valid_std = []
        valid_ewma = []
        valid_garch = []
        
        # DataFrames acumuladores para Auditoria
        audit_prices = pd.DataFrame()
        audit_returns = pd.DataFrame()
        audit_ewma = pd.DataFrame()
        audit_garch = pd.DataFrame()
        
        s_str = start_date.strftime("%Y-%m-%d")
        e_str = end_date.strftime("%Y-%m-%d")
        
        for ticker_raw in tickers:
            ticker = ticker_raw.strip().upper()
            if not ticker.endswith(".SA") and any(char.isdigit() for char in ticker): ticker += ".SA"
            try:
                data = yf.download(ticker, start=s_str, end=e_str, progress=False)
                if data.empty:
                    results[ticker_raw] = {"error": "Sem dados"}
                    continue
                series = None
                if 'Adj Close' in data.columns: series = data['Adj Close']
                elif 'Close' in data.columns: series = data['Close']
                if series is None: continue
                if isinstance(series, pd.DataFrame): series = series.iloc[:, 0]
                
                # Salva Preços para Auditoria
                audit_prices[ticker_raw] = series
                
                returns = np.log(series / series.shift(1)).dropna()
                if len(returns) < 30: continue
                
                # Salva Retornos para Auditoria
                audit_returns[ticker_raw] = returns

                # Cálculos
                std_vol = returns.std() * np.sqrt(252)
                
                ewma_series = returns.ewm(alpha=(1 - 0.94)).std() * np.sqrt(252)
                ewma_vol = ewma_series.iloc[-1]
                audit_ewma[ticker_raw] = ewma_series # Salva Série EWMA

                # GARCH
                garch_vol = None
                try:
                    r_scaled = returns * 100 
                    model = arch_model(r_scaled, vol='Garch', p=1, q=1, rescale=False)
                    res = model.fit(disp='off', show_warning=False)
                    
                    # Forecast D+1
                    calc_garch = np.sqrt(res.forecast(horizon=1).variance.iloc[-1, 0] * 252) / 100
                    
                    # Série Condicional (Histórica) para Auditoria
                    cond_vol_annual = (res.conditional_volatility / 100) * np.sqrt(252)
                    audit_garch[ticker_raw] = cond_vol_annual

                    if calc_garch is not None and not np.isnan(calc_garch):
                        garch_vol = calc_garch
                        valid_garch.append(garch_vol)
                except: 
                    garch_vol = None

                valid_std.append(std_vol)
                valid_ewma.append(ewma_vol)
                
                results[ticker_raw] = {
                    "std_dev": std_vol, 
                    "ewma": ewma_vol, 
                    "garch": garch_vol, 
                    "last_price": float(series.iloc[-1])
                }
            except Exception as e: results[ticker_raw] = {"error": str(e)}
        
        # Gera o Excel de Auditoria se houver dados
        excel_bytes = None
        if not audit_returns.empty:
            excel_bytes = MarketDataService._generate_excel_audit(
                results, audit_prices, audit_returns, audit_ewma, audit_garch
            )

        return {
            "summary": {
                "mean_std": np.mean(valid_std) if valid_std else 0.0, 
                "mean_ewma": np.mean(valid_ewma) if valid_ewma else 0.0, 
                "mean_garch": np.mean(valid_garch) if valid_garch else 0.0,
                "count_valid": len(valid_std)
            }, 
            "details": results,
            "audit_excel": excel_bytes # <--- Payload do arquivo
        }

    # --- MÓDULO DI FUTURO (MANTIDO IGUAL AO ANTERIOR) ---
    @staticmethod
    def gerar_url_di(data_ref: date) -> str:
        if isinstance(data_ref, str):
            try: data_ref = datetime.strptime(data_ref, "%Y-%m-%d").date()
            except: pass
        d_fmt = data_ref.strftime("%d/%m/%Y")
        return f"https://www2.bmf.com.br/pages/portal/bmfbovespa/boletim1/SistemaPregao_excel1.asp?Data={d_fmt}&Mercadoria=DI1&XLS=true"

    @staticmethod
    def converter_vencimento_ref(di_code):
        meses = {"F": "01", "G": "02", "H": "03", "J": "04", "K": "05", "M": "06", 
                 "N": "07", "Q": "08", "U": "09", "V": "10", "X": "11", "Z": "12"}
        di_code = str(di_code).strip().upper()
        if len(di_code) == 3 and di_code[0] in meses:
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
            tabelas_dfs = pd.read_html(response.content, encoding='latin1', decimal=',', thousands='.')

            if len(tabelas_dfs) < 7: return pd.DataFrame()

            df = tabelas_dfs[6]
            df.columns = df.iloc[1]
            df = df.iloc[2:].reset_index(drop=True)
            if df.iloc[-1, 0] is None or pd.isna(df.iloc[-1, 0]): df = df.iloc[:-1]

            mapa_colunas = {
                'VENCTO': 'VENCIMENTO', 
                'VENC.': 'VENCIMENTO',
                'ÚLT. PREÇO': 'ULTIMO PRECO',
                'ULT. PREÇO': 'ULTIMO PRECO',
                'AJUSTE': 'PRECO AJUSTE'
            }
            cols_found = {}
            for c in df.columns:
                c_clean = str(c).strip().upper()
                if c_clean in mapa_colunas: cols_found[c] = mapa_colunas[c_clean]
            df = df.rename(columns=cols_found)

            if 'VENCIMENTO' not in df.columns: return pd.DataFrame()
            col_preco = 'ULTIMO PRECO' if 'ULTIMO PRECO' in df.columns else 'PRECO AJUSTE'
            if col_preco not in df.columns: return pd.DataFrame()

            clean_data = []
            for _, row in df.iterrows():
                try:
                    venc_cod = row['VENCIMENTO']
                    taxa_raw = row[col_preco]
                    if pd.isna(venc_cod) or pd.isna(taxa_raw): continue
                    venc_fmt = MarketDataService.converter_vencimento_ref(venc_cod)
                    if not venc_fmt: continue
                    dt_venc = datetime.strptime(venc_fmt, "%m/%Y").date()
                    
                    if isinstance(taxa_raw, str): taxa_raw = taxa_raw.replace('.', '').replace(',', '.')
                    
                    # Correção Determinística (Padrão B3 4 casas)
                    val = float(taxa_raw) / 100000.0
                    
                    dias_corridos = (dt_venc - reference_date).days
                    if dias_corridos > 0:
                        clean_data.append({
                            "Vencimento_Fmt": venc_fmt,
                            "Vencimento_Str": str(venc_cod),
                            "Vencimento_Data": dt_venc,
                            "Dias_Corridos": dias_corridos,
                            "Taxa": val
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
