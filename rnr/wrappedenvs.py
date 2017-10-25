from gym.envs.classic_control import PendulumEnv

class RestartablePendulumEnv(PendulumEnv):
    def __init__(self,*args,**kwargs):
        super(RestartablePendulumEnv,self).__init__()

    def get_state(self):
        return (self.state,self.last_u)
    def set_state(self,state):
        self.state,self.last_u = state
