import pandas as pd
import numpy as np


def EWMA(s:pd.Series,window:int)->pd.Series:
    """
    Implementação da média movel exponencial
    atributos:
    s: series de preços 
    window: tamanho da janela movel
    return: serie de precos aplicada a media exponencial
    """
    ewma = s.ewm(span=window, adjust=False).mean()
    ewma.iloc[:window] = np.nan
    return ewma


