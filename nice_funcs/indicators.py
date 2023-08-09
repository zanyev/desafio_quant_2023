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


def CreateRandomPrtf(n):   
    """Gera um array de tamanho n+1, onde suas entradas somam 1.
    Método de sorteio em um simplexo. Possui distribuição uniforme.
    int -> array"""

    aleatorio = [0]
    for _ in range(0, n):
        aleatorioi = np.random.rand()   # Gero números entre [0, 1) uniformemente
        aleatorio.append(aleatorioi)    # Os adiciono a lista.

    aleatorio.sort()                    # Ordeno-a
    aleatorio.append(1)

    pesos = np.array(aleatorio[1:]) - np.array(aleatorio[:-1])
    
    return pesos



def MACD(s:pd.Series, media_longa = 26, media_curta = 12)-> [pd.Series, pd.Series, pd.Series]:
    """Recebendo a série de fechamento da ação, retorna 
    MACD = EWMA_26 - EWMA_12
    Signal = EWMA_9(MACD)
    Histograma = Signal - MACD"""

    MACD_ = EWMA(s, media_longa) - EWMA(s, media_curta)
    MACD_s = EWMA(MACD_, 9)
    MACD_h = MACD_ - MACD_s

    return MACD_, MACD_s, MACD_h

