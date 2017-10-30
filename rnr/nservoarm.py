import gym
import numpy as np
import random
from gym import spaces
from gym.utils import seeding
import copy

from math import acos, asin, atan2, cos, pi, sin, sqrt, sqrt


class NServoArmEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 5
    }
    loopcnt=0
    def __init__(self,**kwargs):
        self.height=160
        self.width=320
        self.max_angle=np.pi
        self.max_speed=1
        self.max_torque=1
        self.dt=.05
        self.viewer = None
        self.links=kwargs.get('links',[1.0,1.0])
        self.nsensors=len(self.links)
        sz = np.sum(self.links)
        self.bounds = (-1.1 * sz, 1.1 * sz, -0.1 * sz, 1.1 * sz)
        self.action_space = spaces.Box(low=-self.max_speed, high=self.max_speed, shape=(len(self.links),))

        self.image_goal=kwargs.get('image_goal',None) # NOTE: in this mode a display is required
        self.image_only=kwargs.get('image_only',False) #
        self.bar=kwargs.get('bar',len(self.links)>2)
        self.torquein=kwargs.get('torquein',True) # use input as a torqueue
        self.initial_angles = [tuple([float(x) for x in t]) for t in kwargs.get('initial_angles', [])]
        self.polar = kwargs.get('polar', False)
        self.terminate_on_invalid = kwargs.get('terminate_on_invalid', False)
        self.softpenalties = kwargs.get('softpenalties', False)
        self.hardpenalties = kwargs.get('hardpenalties', False)
        self.d2reward = kwargs.get('d2reward', False)
        self.sinout=kwargs.get('sine',True)
        self.deadband_reward=kwargs.get('deadband_reward',True)
        self.overx=kwargs.get('overx',True)
        self.negreward=kwargs.get('negd',not self.d2reward)
        self.deadband_stop=kwargs.get('deadband_stop',False)
        self.verbose=kwargs.get('verbose',False)
        self.use_random_goals=kwargs.get('use_random_goals',False)
        self.clipping=kwargs.get('clipping',False)
        self.clipreward=kwargs.get('clipreward',True)
        self.reward_scale = 1 / (2 * sz)

        if self.image_goal is not None:
            self.height, self.width = kwargs["image_goal"]
            self.channels = 3
            if self.verbose:
                print("Image observation {}x{}".format(self.height,self.width))
            if self.image_only:
                self.observation_space = spaces.Box(low=0,high=1.0,shape=(self.height,self.width,self.channels))
            else:
                self.high = np.array([self.max_angle] * len(self.links) + [255]*self.width*self.height*self.channels)
                self.low = np.array([self.max_angle] * len(self.links) + [0]*self.width*self.height*self.channels)
                self.observation_space = spaces.Box(low=self.low, high=self.high)
        else:
            self.high = np.array([self.max_angle] * len(self.links) + ([1, 1] if self.bar else [1,1,1]))
            self.observation_space = spaces.Box(low=-np.pi, high=np.pi, shape=(len(self.links)*(1+self.torquein+self.sinout)+(3 if self.bar else 2),))

        self._seed()
        self.linkx=np.zeros_like(self.links)
        self.linky=np.zeros_like(self.links)
        self.linka=np.zeros_like(self.links)
        self.linkv=np.zeros_like(self.linka)

        self.deadband=kwargs.get('goal_size',0.1)
        self.lastdr=0

        self.locked=False
        if 'goals' in kwargs:
            self.goals = [tuple([float(x) for x in t]) for t in kwargs['goals']]
        elif 'ngoals' in kwargs:
            self.random_goals(int(kwargs['ngoals']))
        else:
            self.random_goals(1)
        self.myname=random.randint(0,1000)

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _step(self,u):
        #move
        if self.clipping:
            cu=np.clip(u,-self.max_torque,self.max_torque)
        else:
            cu=u
        for i,l in enumerate(self.links):
            if self.torquein:
                self.linkv[i]+=cu[i]*self.dt*0.1 #
                self.linka[i]+= self.linkv[i] * self.dt
            else:
                self.linkv[i]=cu[i]
                self.linka[i] += self.linkv[i] * self.dt

        if np.any(np.greater(self.linkv,2*self.max_speed)): # stop if grossly overspeed
            self.done=True
        if np.any(np.greater(np.abs(self.linka),20*np.pi)): # stop if grossly overrotating
            self.done=True

        if self.clipping:
            self.linkv=np.clip(self.linkv,-self.max_speed,self.max_speed)
        # find new position of the end effector and distance to goal
        xs, ys, ts,d = self.node_pos()


        # determine reward
        if self.negreward:
            reward = -d
        elif self.d2reward:
            reward = -d**2
        else:
            reward = (self.lastd - d) * self.reward_scale  # reward for progress towards the goal
        self.lastd = d

        reward += -0.001 - np.sum(np.square(u))*0.01 - np.sum(np.square(self.linkv))*0.1   # incentive to get something done and not waste energy
        if self.bar:
            ag=self.goals[self.goalidx][2]
            reward -= (1-(cos(ts[-1]) * cos(ag) + sin(ts[-1]) * sin(ag))) * 0.1

        invalid=False


        rsoft=0
        def softlimit(v,ul,ll=None):
            try:
                if np.isnan(v).any():
                    pass
                if ll==None:
                    v=np.abs(v)
                r=np.where(v>ul,v-ul,0)
                if ll:
                    r+=np.where(v<ll,-(v+ll),0)
                r=np.sum(np.square(r))
            except:
                print("NAN")
            return None if r==0 else r

        r=softlimit(np.array(ys),100000,-0.02)
        if r:
            if self.verbose: print("Below ground {}".format(r))
            rsoft -= r*.1
            invalid=True

        r=softlimit(self.linka,np.pi-0.2)
        if r:
            rsoft -= r*.01
            if self.verbose:print("Rotation limit {}".format(r))
            invalid=True
        if self.overx:
            r=softlimit(u, self.max_torque)
            if r:
                if self.verbose:print("Over torque {}".format(r))
                reward -= r*.1
            r=softlimit(self.linkv,self.max_speed)
            if r:
                if self.verbose:print("Overspeed {}".format(r))
                reward -= r*0.01

        if self.softpenalties:
            reward+=rsoft
        elif self.hardpenalties:
            if invalid:
                reward -= 2

        if invalid:
            if self.verbose:print("Invalid obs={} u={} reward={} ys {} d {} er {} ".format(self._get_obs(), u, reward, ys, d,
                                                                 self.episode_reward))
            if self.terminate_on_invalid:
                self.done=True
        if self.deadband_stop:
            if d < self.deadband:
                if self.deadband_reward:
                    reward += 1
                self.done = True
        else:
            if self.deadband_reward:
                if self.softpenalties and d < self.deadband*2:
                    reward += (self.deadband*2-d)/self.deadband*0.25

        if self.clipreward:
            reward=np.clip(reward,-10,0)

        self.episode_reward+=reward

        if self.done and self.verbose: print("Done    obs={} u={} reward={} ys {} d {} er {} ".format(self._get_obs(),u,reward,ys,d,self.episode_reward))
        NServoArmEnv.loopcnt+=1
        return self._get_obs(), reward, self.done, {
            "goal":self.goals[self.goalidx],
            'bounds':self.bounds,
            'effector':(xs[-1],ys[-1]),
        }

    def get_state(self):
        assert not self.use_random_goals
        return copy.deepcopy((self.linka,self.linkv,self.goalidx,self.done,self.lastd,self.creset,self.episode_reward, self.deadband_count))

    def set_state(self,state):
        self.linka,self.linkv,self.goalidx,self.done,self.lastd,self.creset,self.episode_reward, self.deadband_count=copy.deepcopy(state)
        return self._get_obs()

    def _reset(self):
        self.linka= np.zeros_like(self.linka)
        self.linkv=np.zeros_like(self.linkv)
        self.done=False
        self.creset=True #controller reset

        if self.use_random_goals: # use a new goal each time
            self.random_goals(1)
        self.goalidx=random.randrange(len(self.goals))
        if len(self.initial_angles)>0:
            self.linka=np.array(random.choice(self.initial_angles))
            _, _, _, self.lastd = self.node_pos()
        else:
            while True: #pick a random but valid state
                self.linka = np.random.uniform(-np.pi, np.pi, size=[len(self.links)])
                _,ys,_,self.lastd = self.node_pos()
                if np.all(np.greater_equal(ys[1:],0.2)):break
        self.episode_reward=0
        self.deadband_count=0
        return self._get_obs()

    def _get_obs(self):
        if self.image_only:
            return self._render(mode='rgb_array')
        # normalize the goal
        goal = np.array(self.goals[self.goalidx])
        goal[0]/=np.sum(self.links) #normalize length
        goal[1]/=np.sum(self.links) #normalize length
        if self.bar:
            goal[2]/=np.pi #normalize bar angle
        if self.polar:
            x=goal[0]
            y=goal[1]
            goal[0]=atan2(x,y)/np.pi
            goal[1]=sqrt(x**2+y**2)
        s=[]
        if self.sinout:
            s.append(np.cos(self.linka))
            s.append(np.sin(self.linka))
        else:
            s.append([angle_normalize(self.linka)])
        if self.torquein: s.append(self.linkv)
        if self.image_goal:
            img = self._render(mode='rgb_array')
            s.append(img.flatten())
        else:
            s.append(goal)
        return np.concatenate(s)


    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(self.width,self.height)
            self.viewer.set_bounds(*self.bounds)
            self.pole_transforms=[]

            for i,l in enumerate(self.links):
                rod = rendering.make_capsule(l, 0.2/(i+1))
                rod.set_color(.3, .8, .3)
                transform = rendering.Transform()
                rod.add_attr(transform)
                self.pole_transforms.append(transform)
                self.viewer.add_geom(rod)


            axle = rendering.make_circle(.05)
            axle.set_color(0,0,0)
            self.viewer.add_geom(axle)

            if self.bar:
                goal = rendering.make_capsule(self.deadband*2, self.deadband )
                goal.set_color(1,0,0)
            else:
                goal = rendering.make_circle(self.deadband)
                goal.set_color(1,0,0)
            self.goal_transform = rendering.Transform()
            goal.add_attr(self.goal_transform)
            self.viewer.add_geom(goal)

        for px,x,y,t in zip(self.pole_transforms, *self.node_pos(False)):
            # there is a race condition of some sort, because
            px.set_rotation(t)
            px.set_translation(x,y)

        self.goal_transform.set_translation(self.goals[self.goalidx][0],self.goals[self.goalidx][1])
        if self.bar: self.goal_transform.set_rotation(self.goals[self.goalidx][2])

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def node_pos(self,withd=True):
        xs=[0]
        ys=[0]
        ts=[np.pi/2] # zero is straight up
        for i,l in enumerate(self.links):
            ts.append(ts[-1] + self.linka[i])
            xs.append(xs[-1]+l*cos(ts[-1]))
            ys.append(ys[-1]+l*sin(ts[-1]))
        del ts[0]
        if withd:
            d = sqrt((self.goals[self.goalidx][0] - xs[-1]) ** 2 + (self.goals[self.goalidx][1] - ys[-1]) ** 2)
            return xs,ys,ts,d # xs&ys include end effector
        else:
            return xs,ys,ts

    def random_goals(self,n):
        self.goals=[]
        for i in range(n):
            r = np.sum(self.links)
            r *= np.random.uniform(0.3, 1)
            angle = np.random.uniform(0.2, np.pi-0.2)
            if self.bar:
                gangle=np.random.uniform(-np.pi, np.pi)
                self.goals.append((r * cos(angle), r * sin(angle),gangle))
            else:
                self.goals.append((r * cos(angle),r * sin(angle)))

    def set_goals(self,goals):
        self.goals=goals
        self.reset()
        return

    def get_goal_idx(self):
        return self.goalidx

    def controller(self): # state aware controller
        #first determine goal angles
        if self.creset:
            x,y=self.goals[self.goalidx]
            n = len(self.links)
            assert n in [2]
            r = sqrt(x ** 2 + y ** 2)
            a = atan2(y, x) - pi / 2  # zero is straight up
            l0 = self.links[0]
            l1 = self.links[1]
            a1 = np.pi - acos((l0 ** 2 + l1 ** 2 - r ** 2) / (2 * l0 * l1))
            a0 = -asin(l1 / r * sin(np.pi - a1))
            # switch to always be above
            if x < 0:
                a0 = a + a0
            else:
                a0 = a - a0
                a1 = -a1
            self.goala=np.array([a0,a1])
            self.creset=False

        # now determine torques PID
        lv=np.array(self.linkv)
        ad=self.goala-np.array(self.linka) # goal angle - current angle
        v= ad*3.0- lv*10.0
        v = v *self.max_speed/3/(abs(v)+self.max_speed/3) # soft clip
        vd = v - lv #velocity difference
        t = vd*10
        t = vd*self.max_torque/(abs(vd)+self.max_torque) # soft clip
        return t


def angle_normalize(x): # convert angle to -1 to +1
    return (((x/np.pi+1) % 2) - 1)

