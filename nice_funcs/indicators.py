import pandas as pd
import numpy as np


btc = pd.read_excel(r'C:\Users\Renato\Desktop\desafio_quant\desafio_quant_2023\assets\1d\BTCUSDT.xlsx')

def EWMA(s:pd.Series,window:int)->pd.Series:
    ewma = s.ewm(span=10, adjust=False).mean()
    return ewma
def MACD():
    pass

print(EWMA(btc['Close'],10))