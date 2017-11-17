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
        self.maxv=0.5
        self.maxa=1
        self.maxu=1
        self.ndim=kwargs.get('ndim',1)
        self.goalx=np.array([0]*self.ndim)
        self.bounds=np.array([self.maxx,self.maxv,self.maxx]*self.ndim)
        self.action_space = spaces.Box(low=-self.maxa, high=self.maxa, shape=(self.ndim,))
        self.observation_space = spaces.Box(low=-self.bounds,high=self.bounds)
        self.deadband=0.01
        self.vdeadband=0.10
        self.dt=0.1
        self.x=0
        self.v=0

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self,u):
        u=np.clip(u, -self.maxu, self.maxu)
        self.v+=u*self.dt
        self.v=np.clip(self.v,-self.maxv,self.maxv)
        self.x += self.v*self.dt
        error=np.sum(abs(self.x-self.goalx))
        reward= -np.sum(error**2+0.1*self.v**2+0.01*u[0]**2)
        self.done=( (np.all(error<self.deadband) and np.all(abs(self.v)<self.vdeadband))
                    or np.any(abs(self.x)>self.maxx)
                    or reward<-10)
        if self.done:
            print("done {} {} {} {} {}".format(error,self.goalx,self.x,self.v,u))
        return self._get_obs(), reward, self.done, {}


    def _reset(self):
        self.goalx=np.random.uniform(-self.maxx,self.maxx,size=(self.ndim,))
        self.x=np.random.uniform(-self.maxx,self.maxx,size=(self.ndim,))
        self.v=np.zeros(self.ndim)
        return self._get_obs()

    def _get_obs(self):
        return np.hstack([self.x,self.v,self.goalx])


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
            self.viewer.set_bounds(-1.1 * self.maxx, 1.1 * self.maxx, -1.1 * self.maxx, 1.1 * self.maxx)

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

            if self.ndim == 3:
                goal_shadow = rendering.make_circle(.07)
                goal_shadow.set_color(.5,0.2,0.2)
                self.goal_shadow_transform = rendering.Transform()
                goal_shadow.add_attr(self.goal_shadow_transform)
                self.viewer.add_geom(goal_shadow)

                puck_shadow = rendering.make_circle(.027)
                puck_shadow.set_color(0.2,0.2,0.2)
                self.puck_shadow_transform = rendering.Transform()
                puck_shadow.add_attr(self.puck_shadow_transform)
                self.viewer.add_geom(puck_shadow)


        if self.ndim ==1:
            self.goal_transform.set_translation(self.goalx[0],0)
            self.puck_transform.set_translation(self.x[0],0)
        elif self.ndim == 2:
            self.goal_transform.set_translation(self.goalx[0], self.goalx[1])
            self.puck_transform.set_translation(self.x[0], self.x[1])
        elif self.ndim == 3:
            self.goal_transform.set_translation(self.goalx[0]+self.goalx[2], self.goalx[1]+self.goalx[2])
            self.puck_transform.set_translation(self.x[0]+self.x[2], self.x[1]+self.x[2])
            self.goal_shadow_transform.set_translation(self.goalx[0]+self.goalx[2], self.goalx[1]+self.goalx[2])
            self.puck_shadow_transform.set_translation(self.x[0]+self.x[2], self.x[1]+self.x[2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def get_state(self):
        return (self.x,self.v,self.goalx)
    def set_state(self,state):
        self.x,self.v,self.goalx= state
        return self._get_obs()