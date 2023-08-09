import pandas as pd


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

    sri_vector = np.array(s['rsi'])

    return sri_vector
