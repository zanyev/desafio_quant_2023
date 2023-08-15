# %%
import pandas as pd
import numpy as np
import os
from nice_funcs.indicators import CreateRandomPrtf,EWMA,MACD,RSI,NormalizeWindow
from stable_baselines3 import DDPG ,PPO
from ambiente import TradingEnv
from stable_baselines3.common.noise import NormalActionNoise


    

def GetIndex(*args):
  indicators = [*args]
  index_init = set(indicators[0].index)
  for ind_ in indicators:
    index_init = index_init & set(ind_.index)
  
  idx_date = min(index_init)
  new_index_indicators = []
  for ind_ in indicators:
    ind_['Cash'] = 0 
    new_index_indicators.append(ind_[idx_date:])
  return new_index_indicators
        


# %%
path_diario = './assets/1d/'
ativos = os.listdir(path_diario)

ativosOHLC = {}
for ativo in ativos:
    ativosOHLC[ativo.replace('.xlsx','')] = \
        pd.read_excel(os.path.join(path_diario,ativo),index_col=0)
    

close_prices = {}
for k in ativosOHLC.keys():
  close_prices[k] = ativosOHLC[k].Close


df_fechamento = pd.DataFrame(close_prices).iloc[:-360]
normalized_fech = df_fechamento.apply(lambda row: NormalizeWindow(row)).dropna()
macd = normalized_fech.apply(lambda row: MACD(row)[0]).dropna()
rsi = normalized_fech.apply(lambda row: RSI(row)).dropna()
ewma_diff = normalized_fech.apply(lambda row: EWMA(row,20) - EWMA(row,5)).dropna()
df_fechamento,normalized_fech,macd,rsi,ewma_diff =  GetIndex(df_fechamento,normalized_fech, macd, rsi, ewma_diff)


env = TradingEnv(df_fechamento,[normalized_fech,macd,rsi,ewma_diff])
action_noise = NormalActionNoise(mean=np.zeros(len(df_fechamento.columns)), sigma=0.1 * np.ones(len(df_fechamento.columns)))


save_path = os.path.join('Training', 'Saved Models')
log_path = os.path.join('Training', 'Logs')

env = TradingEnv(df_fechamento,[normalized_fech,macd,rsi,ewma_diff])

model = PPO("MlpPolicy",
            env,
            verbose = 1,
            train_freq=4,
            learning_rate=0.001,
            batch_size=128)

model.learn(total_timesteps=2_000_000,progress_bar=True)

model.save('./Training/Saved Models/trading_2.zip')




