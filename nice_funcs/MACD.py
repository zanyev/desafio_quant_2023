import pandas as pd

from indicators import EWMA


def MACD(s:pd.Series, media_longa = 26, media_curta = 12)-> [pd.Series, pd.Series, pd.Series]:
    """Recebendo a série de fechamento da ação, retorna 
    MACD = EWMA_26 - EWMA_12
    Signal = EWMA_9
    Histograma = Signal - MACD"""

    MACD_ = EWMA(s, media_longa) - EWMA(s, media_curta)
    MACD_s = EWMA(MACD_, 9)
    MACD_h = MACD_s - MACD_

    return MACD_, MACD_s, MACD_h
