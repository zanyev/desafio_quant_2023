
from gymnasium import Env
from gymnasium.spaces import Box , Dict
import numpy as np

class TradingEnv(Env):
    def __init__(self,fechamento,indicadores:list)->None:
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
  
      self.done = False
      self.truncated = False
      self.max_steps = len(self.fechamento) -1

      self.action_space = Box(low=0,
                              high=1,
                              shape=(self.n_actions,),
                              dtype='float32')
      
      # [0,0.2,0.2...] 

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

      #obs = dict(zip(self.nome_ativos,arr.T))
      return arr

    def RewardFunc(self,precos,new_precos,action):
      portfolio_return = sum(((new_precos[:-1] - precos[:-1])/precos[:-1]) * action[:-1])
      reward = portfolio_return*self.dinheiro_final

      return reward

    def step(self,action):
      info = {}
      
      obs = self.CreateObs()
      precos = self.fechamento.iloc[self.idx_]
      self.step_ += 1
      self.idx_ +=1

      action = self.softmax_normalization(action)
      
  
      new_precos = self.fechamento.iloc[self.idx_]
      reward = self.RewardFunc(precos,new_precos,action)
  

      

      self.dinheiro_final += reward

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

      self.step_ = 0
      self.idx_ = 0
      self.dinheiro_final = self.dinheiro_inicial
      self.done = False
      self.truncated = False
      obs = self.CreateObs()

      return obs, {}
    
    def softmax_normalization(self, actions):
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator/denominator
        return softmax_output


    def Define_Observation(self):
      space = {}
      for ativo_name in self.nome_ativos:
        space[ativo_name] = Box(low=-np.inf,high=np.inf,shape = (len(self.indicadores),))
      
      return Dict(space)

    
    



      

