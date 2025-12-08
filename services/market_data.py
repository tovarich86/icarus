import pandas as pd
import numpy as np
import yfinance as yf
from arch import arch_model
from datetime import date, timedelta
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import streamlit as st

class MarketDataService:
    """
    Serviço responsável por buscar dados de mercado (Cotações, Volatilidade e Taxas DI).
    """

    @staticmethod
    def get_historical_volatility(ticker: str, window_years: float = 3.0) -> dict:
        """
        Calcula volatilidade histórica usando yfinance e arch (GARCH/EWMA).
        Baseado na lógica do arquivo 'volatilidade.py'.
        """
        ticker = ticker.strip().upper()
        if not ticker.endswith(".SA"):
            ticker += ".SA"
        
        # Define janela de dados (Dias úteis aproximados)
        days = int(window_years * 252)
        start_date = (date.today() - timedelta(days=window_years*365 + 30))
        
        try:
            # Baixa dados
            data = yf.download(ticker, start=start_date, progress=False)
            if data.empty:
                return {"error": "Ticker não encontrado ou sem dados."}
            
            # Limpeza e Retornos
            # Ajuste para lidar com MultiIndex do yfinance novo ou simples
            adj_close = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
            if isinstance(adj_close, pd.DataFrame):
                adj_close = adj_close.iloc[:, 0]
                
            returns = adj_close.pct_change().dropna()
            
            # Recorte para o período exato solicitado
            returns = returns.tail(days)
            
            if len(returns) < 30:
                return {"error": "Dados insuficientes para cálculo estatístico."}

            # 1. Desvio Padrão Simples (Anualizado)
            std_vol = returns.std() * np.sqrt(252)

            # 2. EWMA (Lambda 0.94 - Padrão RiskMetrics)
            ewma_vol = returns.ewm(alpha=(1 - 0.94)).std().iloc[-1] * np.sqrt(252)

            # 3. GARCH(1,1)
            garch_vol = np.nan
            try:
                # Escala x100 para otimização numérica
                r_scaled = returns * 100 
                model = arch_model(r_scaled, vol='Garch', p=1, q=1, rescale=False)
                res = model.fit(disp='off', show_warning=False)
                forecast = res.forecast(horizon=1)
                # Desescala e anualiza
                garch_vol = np.sqrt(forecast.variance.iloc[-1, 0] * 252) / 100
            except Exception:
                garch_vol = None

            return {
                "std_dev": std_vol,
                "ewma": ewma_vol,
                "garch": garch_vol,
                "last_date": returns.index[-1].strftime('%d/%m/%Y'),
                "samples": len(returns)
            }

        except Exception as e:
            return {"error": f"Erro no processamento: {str(e)}"}

    @staticmethod
    @st.cache_data(ttl=3600) # Cache de 1h para não sobrecarregar B3/Selenium
    def get_risk_free_rate_curve() -> pd.DataFrame:
        """
        Faz o scraping da curva DI Pré da B3.
        Adaptado do 'di_be/app.py' para retornar DataFrame limpo.
        """
        url = "https://www2.bmf.com.br/pages/portal/bmfbovespa/lumis/lum-taxas-referenciais-bmf-ptBR.asp"
        
        try:
            options = Options()
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            
            # Instalação automática do driver para compatibilidade
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=options)
            
            driver.get(url)
            # Espera implícita simples
            driver.implicitly_wait(3)
            
            html = driver.page_source
            driver.quit()
            
            soup = BeautifulSoup(html, 'html.parser')
            tabela = soup.find('table', id='tb_principal1')
            
            if not tabela: return pd.DataFrame()
            
            dados = []
            for linha in tabela.find('tbody').find_all('tr'):
                cols = linha.find_all('td')
                if len(cols) == 3:
                    try:
                        dias = int(cols[0].get_text(strip=True))
                        taxa = float(cols[1].get_text(strip=True).replace(',', '.')) / 100
                        dados.append({'dias': dias, 'anos': dias/252, 'taxa': taxa})
                    except:
                        continue
            
            return pd.DataFrame(dados).sort_values('dias')
            
        except Exception as e:
            st.error(f"Erro ao buscar DI: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def interpolate_di_rate(target_years: float, curve_df: pd.DataFrame) -> float:
        """
        Interpola linearmente a taxa DI para o prazo exato (Duration) da opção.
        """
        if curve_df.empty: return 0.1075 # Fallback Selic Meta (exemplo)
        
        # Extrapolação Flat nas pontas
        if target_years <= curve_df['anos'].min():
            return curve_df.iloc[0]['taxa']
        if target_years >= curve_df['anos'].max():
            return curve_df.iloc[-1]['taxa']
            
        # Interpolação Linear
        return np.interp(target_years, curve_df['anos'], curve_df['taxa'])
