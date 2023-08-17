import pandas as pd
import numpy as np


def SharpeRatio(retorno:pd.Series, window=365)->pd.Series:
    """Recebendo a série que contém os retornos da carteira, retorna o índice Sharpe dela.
    sharpe = (Retorno - RiskFree)/std(Retorno)"""
    
    std_retorno = retorno.std()

    # Vamos calcular o retorno do bond de 10 anos dos EUA. 08/08/2018 até 08/08/2023
    path='../assets/United States 10-Year Bond Yield Historical Data.csv'
    close_price_free=pd.read_csv(path)['Close']
    retorno_riskfree = (close_price_free[1:] - close_price_free[:-1]) / close_price_free[:-1]

    indice_sharpe = (np.mean(retorno) - np.mean(retorno_riskfree))/std_retorno

    return indice_sharpe
