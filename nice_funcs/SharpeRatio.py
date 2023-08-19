import pandas as pd
import numpy as np



def SharpeRatio(retorno:pd.Series,risk_free_rate:float, window=20)->pd.Series:
    """Recebendo a série que contém os retornos da carteira, retorna o índice Sharpe dela.
    sharpe = (Retorno - RiskFree)/std(Retorno)"""
    
    mean_returns = retorno.rolling(window=window).mean()
    mean_std = retorno.rolling(window=window).std()

    sharpe = (mean_returns-risk_free_rate)/mean_std

    # Vamos calcular o retorno do bond de 10 anos dos EUA. 08/08/2018 até 08/08/2023
    return sharpe
