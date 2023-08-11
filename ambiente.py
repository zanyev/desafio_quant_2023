
from stable_baselines3.common import env_checker
from gymnasium import Env
from gymnasium.spaces import Box

class TradingEnv(Env):
    def __init__(self,prtf_inicial_value:float)->None:
      super().__init__()
      self.prtf_inicial_value = prtf_inicial_value
      self.positions_active = {}
      self.step_ = 0
      self.prtf_final = prtf_inicial_value

    
    



      

