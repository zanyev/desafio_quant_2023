
from gymnasium import Env
from gymnasium.spaces import Box , Dict
import numpy as np
import pandas as pd
from nice_funcs.indicators import SharpeRatio

class TradingEnv(Env):
    def __init__(self,fechamento,indicadores:list,risk_free_rate:float,objetive:str,long_only:bool)->None:
      super().__init__()
      self.fechamento = fechamento
      self.idx_ = 0
      self.step_ = 0
      self.indicadores = indicadores
      self.dinheiro_inicial = 1000
      self.dinheiro_final = self.dinheiro_inicial
      self.n_indicadores = len(self.indicadores)
      self.n_ativos = len(self.fechamento.columns)
      self.n_actions = self.n_ativos
      self.nome_ativos = self.fechamento.columns
      self.risk_free_rate = risk_free_rate
      self.done = False
      self.truncated = False
      self.max_steps = len(self.fechamento) -1
      self._step_portfolio = [self.dinheiro_inicial]
      self.objetive = objetive

      if long_only:
        self.action_space = Box(low=0,
                                high=1,
                                shape=(self.n_actions,),
                                dtype='float32')
      else:
        self.action_space = Box(low=-1,
                                high=1,
                                shape=(self.n_actions,),
                                dtype='float32')

      
      self.observation_space = Box(low=-np.inf,
                                   high=np.inf,
                                   shape=(self.n_indicadores * self.n_ativos,),
                                   dtype='float32')
 
    def CreateObs(self):
      arr = []
      idx_ = self.idx_

      for ind in self.indicadores:
        arr.append(ind.iloc[idx_].values)
      arr = np.array(arr,dtype='float32')
      arr = arr.flatten()
      return arr
  

    def PNL(self,precos,new_precos,action):
      portfolio_return = sum(((new_precos - precos)/precos) * action)
      reward = portfolio_return*self.dinheiro_final
      return reward
    
    def MovingSharpe(self):
      returns = self.CalculatePrtfReturns()
      sharpe = SharpeRatio(returns,self.risk_free_rate).fillna(0)
      return sharpe.iloc[-1]

    def step(self,action):
      action = self.NormalizeAction(action)
      info = {}
      obs = self.CreateObs()
    
      precos = self.fechamento.iloc[self.idx_]
      self.step_ += 1
      self.idx_ +=1
      new_precos = self.fechamento.iloc[self.idx_]


      
      pnl = self.PNL(precos,new_precos,action)
  
      

      self.dinheiro_final += pnl
      self._step_portfolio.append(self.dinheiro_final)

      if self.objetive  == 'r':
        reward = pnl
      elif self.objetive == 's':
        reward = self.MovingSharpe()
      else:
        raise(TypeError)
      
      
      condicao_1 = (self.dinheiro_final/self.dinheiro_inicial)
      condicao_2 = self.step_ >= self.max_steps
    
      if condicao_1 <= 0.1: # caiu 90% = done
        self.done = True

      if condicao_2:
        self.truncated = True  #acabou os dias de treinamento = done

      return obs,reward,self.done,self.truncated,info


    def reset(self,seed=0):
      print('*'*50)
      print('Retorno',round(self.dinheiro_final,2))


      if self.step_ != 0:
        print("Sharpe", round(self.CalculatePrtfReturns().mean()*np.sqrt(252),4))

      self.step_ = 0
      self.idx_ = 0
      self.dinheiro_final = self.dinheiro_inicial
      self.done = False
      self.truncated = False
      self._step_portfolio = [self.dinheiro_inicial]
      obs = self.CreateObs()

      return obs, {}
    
    
    def NormalizeAction(self, action):
      if all(action) == 0:
        action = 1/len(action)
      else:
        action = action/sum(np.abs(action))
      return action


    def CalculatePrtfReturns(self):
      returns = pd.Series(self._step_portfolio).pct_change()[1:]

      return returns

    
    





      

