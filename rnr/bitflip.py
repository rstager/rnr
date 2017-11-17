import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

class BitflipEnv(gym.Env):
    metadata = {
        'render.modes' : ['human'],
    }
    loopcnt=0
    def __init__(self,bits=10,**kwargs):
        self.bits = bits
        self.max = 2**bits
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=np.array([0,0,0]), high=np.array([self.max,self.bits,self.max]))
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self, u):
        if u:
            self.state[0]|=2^self.state[2]
        self.state[2]+=1
        costs=1 if self.state[0]!=self.state[1] and self.state[2]!=self.bits else 0
        done= self.state[2]==self.bits
        return self._get_obs(), -costs, done, {}

    def _reset(self):
        self.state = np.array([0,np.random.randint(0,self.max+1),0])
        return self._get_obs()

    def _get_obs(self):
        return np.array(self.state)

    def _render(self, mode='human', close=False):
        print("{:b}".format(self.state[0]))