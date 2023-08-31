
import pandas as pd
import numpy as np
import os
from nice_funcs.indicators import CreateRandomPrtf,EWMA,MACD,RSI,NormalizeWindow,MDD
from stable_baselines3 import DDPG ,PPO
from ambiente import TradingEnv
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback


periodo_permitido = ('2009-12-30','2022-12-30')

    

def GetIndex(*args):
  indicators = [*args]
  index_init = set(indicators[0].index)
  for ind_ in indicators:
    index_init = index_init & set(ind_.index)
  
  idx_date = min(index_init)
  new_index_indicators = []
  for ind_ in indicators:
    #ind_['Cash'] = 0 
    new_index_indicators.append(ind_[idx_date:])
  return new_index_indicators
        

path_diario = './assets/1d/'
file = './assets/United States 10-Year Bond Yield Historical Data.csv'

close_price_free = pd.read_csv(file).Price.loc[periodo_permitido[0]:periodo_permitido[1]]
risk_free_rate = close_price_free.mean()/100
daily_risk_free = (risk_free_rate+1)**(1/252) -1

ativos = os.listdir(path_diario)
ativosOHLC = {}
for ativo in ativos:
    ativosOHLC[ativo.replace('.xlsx','')] = \
        pd.read_excel(os.path.join(path_diario,ativo),index_col=0)
    

close_prices = {}
for k in ativosOHLC.keys():
  close_prices[k] = ativosOHLC[k].Close.loc[periodo_permitido[0]:periodo_permitido[1]]


df_fechamento = pd.DataFrame(close_prices).iloc[:-360]

normalized_fech = df_fechamento.apply(lambda row: NormalizeWindow(row)).shift(1).dropna()
macd = df_fechamento.apply(lambda row: MACD(row)[0]).shift(1).dropna()
rsi = df_fechamento.apply(lambda row: RSI(row)).shift(1).dropna()
ewma_diff = df_fechamento.apply(lambda row: EWMA(row,20) - EWMA(row,5)).shift(1).dropna()
ddd = df_fechamento.apply(lambda row: MDD(row,window=26)[0]).shift(1).dropna()
mdd = df_fechamento.apply(lambda row: MDD(row,window=26)[0]).rolling(window=26).min().shift(1).dropna()
df_fechamento,normalized_fech,macd,rsi,ewma_diff,ddd,mdd =  GetIndex(df_fechamento,normalized_fech, macd, rsi, ewma_diff,ddd,mdd)


for val in [df_fechamento,*[normalized_fech,macd,rsi,ewma_diff,ddd,mdd]]:
  print(len(val))

env = TradingEnv(df_fechamento,[normalized_fech,macd,rsi,ewma_diff,ddd,mdd],daily_risk_free,'s',long_only=False)

print(df_fechamento)

save_path = os.path.join('Training', 'Saved Models')
log_path = os.path.join('Training', 'Logs')
eval_callback = EvalCallback(env, best_model_save_path=log_path,
                             log_path=log_path,
                             deterministic=False, render=False)


model = PPO("MlpPolicy",
            env,
            batch_size=128,
            verbose=1,
            tensorboard_log="./Training/Logs/tensor_board_logs")

model.learn(total_timesteps=2_000_000,progress_bar=True,callback=eval_callback)

model.save('./Training/Saved Models/trading_sharpe_4.zip')




