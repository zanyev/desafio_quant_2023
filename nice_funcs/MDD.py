import pandas as pd


def MDD(s:pd.Series, window = 252)->pd.Series:
    """Recebendo a série do close da ação, retorna o maximum drawmdown
    Pretendo calcular o MDD assim:
    Daily_DD = value_today/max_value_window - 1
    MDD = Daily_DD.min()"""

    Max_value = s.rolling(window = window, min_periods=1).max()
    Daily_DD = s/Max_value - 1
    MDD = Daily_DD.min()

    return Daily_DD, MDD
