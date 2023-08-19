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

def NormalizeWindow(s:pd.Series,window=20)->pd.Series:
    """
    Normaliza atraves do calculo do z da normal
    atributos:
    s:serie de preços
    return: serie de precos normalizada
    """
    media = s.rolling(window=window).mean()
    desvio = s.rolling(window=window).std()

    normalized = s.subtract(media).divide(desvio)

    return normalized


def CreateRandomPrtf(n):   
    """Gera um array de tamanho n+1, onde suas entradas somam 1.
    Método de sorteio em um simplexo. Possui distribuição uniforme.
    int -> array"""

    aleatorio = [0]
    for _ in range(0, n-1):
        aleatorioi = np.random.rand()   # Gero números entre [0, 1) uniformemente
        aleatorio.append(aleatorioi)    # Os adiciono a lista.

    aleatorio.sort()                    # Ordeno-a
    aleatorio.append(1)

    pesos = np.array(aleatorio[1:]) - np.array(aleatorio[:-1])
    
    return pesos




def MACD(s:pd.Series, media_longa = 26, media_curta = 12)-> [pd.Series, pd.Series, pd.Series]:
    """Recebendo a série de fechamento da ação, retorna 
    MACD = EWMA_12 - EWMA_26
    Signal = EWMA_9
    Histograma = Signal - MACD"""

    MACD_ = EWMA(s, media_curta) - EWMA(s, media_longa) 
    MACD_s = EWMA(MACD_, 9)
    MACD_h = MACD_s - MACD_

    return MACD_, MACD_s, MACD_h



def RSI(s:pd.Series, window:int=14)->pd.Series:
    """Recebendo a série de fechamento da ação, retorna
    RSI = 100 - 100/(1 + average_gain/average_loss)"""
    s = pd.DataFrame(s)

    s['diff'] = s.diff(1)   # Calcula a diferença do valor atual com o valor anterior
    s['gain'] = s['diff'].clip(lower=0)  # Pega todos os valores >= 0
    s['loss'] = s['diff'].clip(upper=0).abs()  # Pega todos os valores <= 0
                                    # A função coloca 0 onde os valores não estão dentro das condições.

    s['avg_gain'] = s['gain'].rolling(window=window, min_periods=window).mean()[:window+1]    # Calcula a média da [perda, ganho] dos
    s['avg_loss'] = s['loss'].rolling(window=window, min_periods=window).mean()[:window+1]    # 14 primeiros dias.

    for i, _ in enumerate(s['avg_gain'].iloc[window+1:]):  # Calcula o ganho médio dos dias (apartir do dia 14), usando a média WSM
        next_avg_gain = (s['avg_gain'].iloc[i + window] * (window - 1) + s['gain'].iloc[i + window + 1]) / window
        s['avg_gain'].iloc[i + window + 1] = next_avg_gain

    for i, _ in enumerate(s['avg_loss'].iloc[window+1:]):  # Calcula a perda média dos dias (apartir do dia 14), usando a média WSM
        next_avg_loss = (s['avg_loss'].iloc[i + window] * (window - 1) + s['loss'].iloc[i + window + 1]) / window
        s['avg_loss'].iloc[i + window + 1] = next_avg_loss

    s['rs'] = s['avg_gain']/s['avg_loss']
    s['rsi'] = 100 - 100/(1 + s['rs'])

    rsi_vector = np.array(s['rsi'])

    return rsi_vector


def MDD(s:pd.Series, window = 252)->pd.Series:
    """Recebendo a série do close da ação, retorna o maximum drawmdown
    Pretendo calcular o MDD assim:
    Daily_DD = value_today/max_value_window - 1
    MDD = Daily_DD.min()"""

    Max_value = s.rolling(window = window, min_periods=1).max()
    Daily_DD = s/Max_value - 1
    MDD = Daily_DD.min()

    return Daily_DD, MDD



def SharpeRatio(retorno:pd.Series,risk_free_rate:float, window=20)->pd.Series:
    """Recebendo a série que contém os retornos da carteira, retorna o índice Sharpe dela.
    sharpe = (Retorno - RiskFree)/std(Retorno)"""
    
    mean_returns = retorno.rolling(window=window).mean()
    mean_std = retorno.rolling(window=window).std()

    sharpe = (mean_returns-risk_free_rate)/mean_std

    # Vamos calcular o retorno do bond de 10 anos dos EUA. 08/08/2018 até 08/08/2023
    return sharpe