import gym
import numpy as np
import random
from gym import spaces
from gym.utils import seeding


class SliderEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 5
    }
    loopcnt=0
    def __init__(self,**kwargs):
        self.height=160
        self.width=320
        self.viewer=None
        self.maxx=1
        self.maxv=1
        self.maxa=1
        self.goalx=0
        self.bounds=np.array([-self.maxx,-self.maxv])
        self.action_space = spaces.Box(low=-self.maxa, high=self.maxa, shape=(1,))
        self.observation_space = spaces.Box(low=-self.bounds,high=self.bounds)
        self.deadband=0.01
        self.vdeadband=0.01
        self.dt=0.1
        self.x=0
        self.v=0

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self,u):
        self.v+=u[0]*self.dt
        self.x += self.v*self.dt
        error=abs(self.x-self.goalx)
        reward=-error**2-self.v**2
        self.done= (error<self.deadband and abs(self.v)<self.vdeadband) or abs(self.x)>self.maxx
        return self._get_obs(), reward, self.done, {}


    def _reset(self):
        self.goalx=random.uniform(-self.maxx,self.maxx)
        self.goalx=0.0
        self.x=random.uniform(-self.maxx,self.maxx)
        self.v=0
        return self._get_obs()

    def _get_obs(self):
        return np.array([self.x,self.v])


    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(self.width,self.height)
            sz=self
            self.viewer.set_bounds(-1.1 * self.maxx, 1.1 * self.maxx, -1.0, 1.0)

            goal = rendering.make_circle(.05)
            goal.set_color(1, 0, 0)
            self.goal_transform = rendering.Transform()
            goal.add_attr(self.goal_transform)
            self.viewer.add_geom(goal)

            puck = rendering.make_circle(.025)
            puck.set_color(0,0,0)
            self.puck_transform=rendering.Transform()
            puck.add_attr(self.puck_transform)
            self.viewer.add_geom(puck)



        self.goal_transform.set_translation(self.goalx,0)
        self.puck_transform.set_translation(self.x,0)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def get_state(self):
        return (self.x,self.v,self.goalx)
    def set_state(self,state):
        self.x,self.v,self.goalx= state