import pandas as pd
import numpy as np


def SharpeRatio(retorno:pd.Series, window=252)->pd.Series:
    """Recebendo a série que contém os retornos da carteira, retorna o índice Sharpe dela.
    sharpe = (Retorno - RiskFree)/std(Retorno)"""

    Retorno_movel = retorno.rolling(window=window, min_periods=1)   # Faz a janela móvel dos retornos
    std_movel = Retorno_movel.std()                                 # Desvio padrão das janelas
    
    riskfree = 0

    indice_sharpe = (Retorno_movel - riskfree)/std_movel

    return indice_sharpe