import pandas as pd


def MACD(s:pd.Series)-> [pd.Series, pd.Series, pd.Series]:
    """Recebendo a série de fechamento da ação, retorna 
    MACD = EWMA_12 - EWMA_26
    Signal = EWMA_9
    Histograma = Signal - MACD"""

    MACD = EWMA(s, 12) - EWMA(s, 26)
    MACD_s = EWMA(s, 9)
    MACD_h = MACD_s - MACD

    return MACD, MACD_s, MACD_h
