import pandas as pd


def RSI(s:pd.Series, window:int=14)->pd.Series:
    """Recebendo a série de fechamento da ação, retorna
    RSI = 100 - 100/(1 + average_gain/average_loss)"""
    s = s.T
    diferenca = s.diff(1)   # Calcula a diferença do valor atual com o valor anterior
    gain = diferenca.clip(lower=0)  # Pega todos os valores >= 0
    loss = diferenca.clip(upper=0)  # Pega todos os valores <= 0
                                    # Essa função preenche com 0, ao invés ficar mudando o len do vetor final.

    avg_gain = gain.rolling(window=window, min_periods=window).mean()[:window+1]    # Calcula as média das [perdas, ganhos]
    avg_loss = loss.rolling(window=window, min_periods=window).mean()[:window+1]    # usando a média aritmética

    for i, row in enumerate(avg_gain.iloc[window+1:]):  # Calcula o ganho médio, usando a implementação WSM
        avg_gain.iloc[i + window + 1] = (avg_gain.iloc[i + window] * (window - 1) + gain.iloc[i + window + 1]) / window
