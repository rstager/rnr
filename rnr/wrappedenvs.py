from gym.envs.classic_control import PendulumEnv
from gym.envs.classic_control import Continuous_MountainCarEnv

class RestartablePendulumEnv(PendulumEnv):
    def __init__(self,*args,**kwargs):
        super(RestartablePendulumEnv,self).__init__()

    def get_state(self):
        return (self.state,self.last_u)
    def set_state(self,state):
        self.state,self.last_u = state
        return self._get_obs()

class RestartableContinuousMountainCartEnv(Continuous_MountainCarEnv):
    def __init__(self,*args,**kwargs):
        super(RestartableContinuousMountainCartEnv,self).__init__()

    def get_state(self):
        return (self.state)
    def set_state(self,state):
        self.state = state
        return self._get_obs()