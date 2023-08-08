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


def Normalize(s:pd.Series)->pd.Series:
    """
    Normaliza atraves do calculo do z da normal
    atributos:
    s:serie de preços
    return: serie de precos normalizada
    """
    normalized = (s - s.mean())/s.std()

    return normalized


