import gym
import numpy as np
import random
from gym import spaces
from gym.utils import seeding
import numbers

class SliderEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 5
    }
    loopcnt=0
    def __init__(self,**kwargs):
        self.height=kwargs.get('height',320)
        self.width=kwargs.get('width',320)
        self.viewer=None
        self.maxv=0.5
        self.maxa=1
        self.maxu=1
        self.ndim=kwargs.get('ndim',1)
        self.maxx=kwargs.get('maxx',1)
        self.minx=kwargs.get('minx',[x*-1 for x in self.maxx] if hasattr(self.maxx, "__iter__") else -self.maxx)
        if isinstance(self.maxx, numbers.Number):
            self.maxx = [self.maxx] * self.ndim
        if isinstance(self.minx, numbers.Number):
            self.minx = [self.minx] * self.ndim
        if isinstance(self.maxv, numbers.Number):
            self.maxv = [self.maxv] * self.ndim
        self.maxx=np.array(self.maxx)
        self.minx=np.array(self.minx)
        self.maxv=np.array(self.maxv)
        self.wrap=kwargs.get('wrap',False)
        self.goalx=np.array([0]*self.ndim)
        self.ubounds=np.hstack([self.maxx,self.maxv,self.maxx]) #,np.array([self.maxx,self.maxv,self.maxx]*self.ndim)
        self.lbounds = np.hstack([self.minx, self.maxv, self.minx])
        #self.lbounds=np.array([self.minx,-self.maxv,self.minx]*self.ndim)
        self.action_space = spaces.Box(low=-self.maxa, high=self.maxa, shape=(self.ndim,))
        self.image_goal=kwargs.get('image_goal', False)
        if self.image_goal:
            self.height,self.width=self.image_goal
        if self.image_goal:
            self.observation_space = spaces.Box(low=0, high=1.0, shape=(self.height, self.width, 3))
        else:
            self.observation_space = spaces.Box(low=self.lbounds,high=self.ubounds)
        self.deadband=0.1
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

        if not self.wrap:
            cx = np.clip(self.x, self.minx, self.maxx)
            self.hit_limit = np.not_equal(cx,self.x).any()
            self.v=np.where(np.equal(cx,self.x),self.v,0)
            self.x=cx
        else:
            self.x=np.mod(self.x-self.minx,self.maxx-self.minx)+self.minx
            self.hit_limit=False
        # for i in range(self.ndim):
        #     if self.x[i]>self.maxx:
        #         if self.wrap:
        #             self.x[i]-=(self.maxx-self.minx)
        #         else:
        #             self.x[i]=self.maxx
        #             self.v[i]=0
        #         self.hit_limit=True
        #     elif self.x[i] < self.minx:
        #         if self.wrap:
        #             self.x[i]+=(self.maxx-self.minx)
        #         else:
        #             self.x[i]=self.minx
        #             self.v[i]=0
        #         self.hit_limit=True
        error=abs(self.x-self.goalx)
        reward= -np.sum(error**2+0.1*self.v**2+0.01*u[0]**2) - 1.0
        self.done=(np.all(error<self.deadband) and np.all(abs(self.v)<self.vdeadband))
        if self.done:
            reward=0.0
        self.nsteps+=1
        # if self.done:
        #     print("done steps {:4} reward={} d2={} goal {} x={} v={} u={}".format(self.nsteps,reward,error,self.goalx,self.x,self.v,u))
        return self._get_obs(), reward, self.done, {}


    def _reset(self):
        self.goalx=np.random.uniform(self.minx,self.maxx,size=(self.ndim,))
        self.x=np.random.uniform(self.minx,self.maxx,size=(self.ndim,))
        self.v=np.zeros(self.ndim)
        self.nsteps=0
        return self._get_obs()

    def _get_obs(self):
        if self.image_goal:
            return self._render(mode='rgb_array')
        else:
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
            oversize= 1.1
            self.viewer.set_bounds(oversize * self.minx, oversize * self.maxx, oversize * self.minx, oversize * self.maxx)

            if self.ndim<3:
                self.goal = rendering.make_circle(.1)
            else:
                self.goal = rendering.make_polygon([(.1,0), (-.1, .05), (-.1, -.05), (.1,0)])
            self.goal.set_color(1, 0, 0)
            self.goal_transform = rendering.Transform()
            self.goal.add_attr(self.goal_transform)
            self.viewer.add_geom(self.goal)

            if self.ndim<3:
                puck = rendering.make_circle(.05)
            else:
                puck = rendering.make_polygon([(.1,0), (-.1, .05), (-.1, -.05), (.1,0)])
            puck.set_color(0,0,0)
            self.puck_transform=rendering.Transform()
            puck.add_attr(self.puck_transform)
            self.viewer.add_geom(puck)



        if self.ndim ==1:
            self.goal_transform.set_translation(self.goalx[0],0)
            self.puck_transform.set_translation(self.x[0],0)
        elif self.ndim >1:
            self.goal_transform.set_translation(self.goalx[0], self.goalx[1])
            self.puck_transform.set_translation(self.x[0], self.x[1])
        if self.ndim == 3:
            self.goal_transform.set_rotation(self.goalx[2]*np.pi)
            self.puck_transform.set_rotation(self.x[2]*np.pi)

        if np.all(abs(self.x-self.goalx)<self.deadband):
            self.goal.set_color(0,1, 0)
        else:
            self.goal.set_color(1,0, 0)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def get_state(self):
        return (self.x,self.v,self.goalx)
    def set_state(self,state):
        self.x,self.v,self.goalx= state
        return self._get_obs()