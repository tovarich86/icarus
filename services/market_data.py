import pandas as pd
import numpy as np
import yfinance as yf
from arch import arch_model
from datetime import date, datetime, timedelta
# CORREÇÃO AQUI: Adicionado 'Any' na importação
from typing import List, Dict, Optional, Union, Any
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import streamlit as st

class MarketDataService:
    """
    Serviço robusto para dados de mercado (Peer Group Volatility & DI Curve).
    """

    @staticmethod
    def get_peer_group_volatility(
        tickers: List[str], 
        start_date: date, 
        end_date: date
    ) -> Dict[str, Any]:
        """
        Calcula a volatilidade de um grupo de empresas (Peer Group).
        Retorna a média do grupo e os detalhes individuais.
        """
        results = {}
        valid_vols_std = []
        valid_vols_ewma = []
        valid_vols_garch = []

        # Garante datas no formato string para o yfinance
        s_str = start_date.strftime("%Y-%m-%d")
        e_str = end_date.strftime("%Y-%m-%d")

        for ticker_raw in tickers:
            ticker = ticker_raw.strip().upper()
            # Adiciona sufixo .SA se for numérico (ex: VALE3 -> VALE3.SA)
            if not ticker.endswith(".SA") and any(char.isdigit() for char in ticker):
                ticker += ".SA"
            
            try:
                # Download (progress=False para não sujar o log)
                data = yf.download(ticker, start=s_str, end=e_str, progress=False)
                
                if data.empty:
                    results[ticker_raw] = {"error": "Sem dados"}
                    continue

                # Tratamento para colunas (yfinance v0.2+ pode retornar MultiIndex)
                adj_close = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
                if isinstance(adj_close, pd.DataFrame):
                    adj_close = adj_close.iloc[:, 0]
                
                # Cálculo de Retornos Logarítmicos
                returns = np.log(adj_close / adj_close.shift(1)).dropna()
                
                if len(returns) < 30:
                    results[ticker_raw] = {"error": "Amostra insuficiente (<30 dias)"}
                    continue

                # 1. Desvio Padrão (Simples)
                std_vol = returns.std() * np.sqrt(252)
                valid_vols_std.append(std_vol)

                # 2. EWMA (RiskMetrics lambda=0.94)
                ewma_vol = returns.ewm(alpha=(1 - 0.94)).std().iloc[-1] * np.sqrt(252)
                valid_vols_ewma.append(ewma_vol)

                # 3. GARCH(1,1)
                garch_vol = np.nan
                try:
                    # Escala x100 para estabilidade numérica do otimizador
                    r_scaled = returns * 100 
                    model = arch_model(r_scaled, vol='Garch', p=1, q=1, rescale=False)
                    res = model.fit(disp='off', show_warning=False)
                    forecast = res.forecast(horizon=1)
                    # Desescala e anualiza
                    garch_vol = np.sqrt(forecast.variance.iloc[-1, 0] * 252) / 100
                    valid_vols_garch.append(garch_vol)
                except:
                    garch_vol = None

                results[ticker_raw] = {
                    "std_dev": std_vol,
                    "ewma": ewma_vol,
                    "garch": garch_vol,
                    "count": len(returns),
                    "last_price": adj_close.iloc[-1]
                }

            except Exception as e:
                results[ticker_raw] = {"error": str(e)}

        # Cálculo das Médias do Peer Group
        summary = {
            "mean_std": np.mean(valid_vols_std) if valid_vols_std else 0.0,
            "mean_ewma": np.mean(valid_vols_ewma) if valid_vols_ewma else 0.0,
            "mean_garch": np.mean(valid_vols_garch) if valid_vols_garch else 0.0,
            "valid_tickers": len(valid_vols_std)
        }

        return {"summary": summary, "details": results}

    @staticmethod
    @st.cache_data(ttl=3600)
    def get_di_curve(reference_date: date) -> pd.DataFrame:
        """
        Busca a curva de juros (DI x Pré) da B3 para uma data específica.
        URL dinâmica baseada na data de referência.
        """
        d_str = reference_date.strftime('%d/%m/%Y')
        d1_str = reference_date.strftime('%Y%m%d')
        
        url = (
            f"https://www2.bmf.com.br/pages/portal/bmfbovespa/lumis/lum-taxas-referenciais-bmf-ptBR.asp"
            f"?Data={d_str}&Data1={d1_str}&slcTaxa=PRE"
        )
        
        driver = None
        try:
            options = Options()
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)
            
            driver.get(url)
            driver.implicitly_wait(3)
            html = driver.page_source
            
            soup = BeautifulSoup(html, 'html.parser')
            
            tabela = soup.find('table', id='tb_principal1')
            if not tabela:
                return pd.DataFrame()
            
            dados = []
            for linha in tabela.find('tbody').find_all('tr'):
                cols = linha.find_all('td')
                if len(cols) == 3:
                    try:
                        dias = int(cols[0].get_text(strip=True))
                        taxa = float(cols[1].get_text(strip=True).replace(',', '.')) / 100
                        anos = dias / 252
                        dados.append({'dias': dias, 'anos': anos, 'taxa': taxa})
                    except:
                        continue
            
            return pd.DataFrame(dados).sort_values('dias')

        except Exception as e:
            # Em caso de erro no Selenium, não quebra a app, apenas loga e retorna vazio
            print(f"Erro Selenium: {str(e)}")
            return pd.DataFrame()
        finally:
            if driver:
                driver.quit()

    @staticmethod
    def interpolate_di_rate(target_years: float, curve_df: pd.DataFrame) -> float:
        """
        Interpola linearmente a taxa para um prazo alvo (em anos).
        """
        if curve_df.empty: return 0.1075 # Fallback
        
        if target_years <= curve_df['anos'].min():
            return curve_df.iloc[0]['taxa']
        if target_years >= curve_df['anos'].max():
            return curve_df.iloc[-1]['taxa']
            
        return float(np.interp(target_years, curve_df['anos'], curve_df['taxa']))
