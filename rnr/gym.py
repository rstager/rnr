import gym
from math import pi
def rnrenvs():
    gym.envs.register(
        id='NServoArm-v0',
        entry_point='rnr.nservoarm:NServoArmEnv',
        max_episode_steps=500,
        kwargs={#'ngoals': 10,  # 'image_goal':(160,320)
                # 'goals': [(1, 1)],
                # 'initial_angles': [[0, 0]],
                'use_random_goals':True,
                }
    )
    gym.envs.register(
        id='NServoArm-v1',
        entry_point='rnr.nservoarm:NServoArmEnv',
        max_episode_steps=500,
        kwargs={#'ngoals': 10,  # 'image_goal':(160,320)
                'goals': [(1, 1)],
                # 'initial_angles': [[0, 0]],
                'deadband_reward': False,
                'deadband_stop':True,
                }
    )
    gym.envs.register(
        id='NServoArm-v3',
        entry_point='rnr.nservoarm:NServoArmEnv',
        max_episode_steps=500,
        kwargs={#'ngoals': 10,  # 'image_goal':(160,320)
                'goals': [(1, 1)],
                'links': [1,.7,.3],
                # 'initial_angles': [[0, 0]],
                'deadband_reward': False,
                'deadband_stop':True,
                }
    )
    gym.envs.register(
        id='NServoArm-v4',
        entry_point='rnr.nservoarm:NServoArmEnv',
        max_episode_steps=500,
        kwargs={
                'goals': [(1, 1)],
                'links': [1,.7,.3],
                'd2reward':True,
                'deadband_reward': False,
                'deadband_stop':True,
                }
    )
    gym.envs.register(
        id='NServoArm-v5',
        entry_point='rnr.nservoarm:NServoArmEnv',
        max_episode_steps=500,
        kwargs={#'ngoals': 10,  # 'image_goal':(160,320)
                'goals': [(1, 1, 0),(-1,1,pi/2),(0,1,0)],
                'links': [1,.7,.3],
                # 'initial_angles': [[0, 0]],
                'deadband_reward': False,
                'deadband_stop':True,
                }
    )
    gym.envs.register(
        id='NServoArmImageOnly-v0',
        entry_point='rnr.nservoarm:NServoArmEnv',
        max_episode_steps=500,
        kwargs={#'ngoals': 10,  # 'image_goal':(160,320)
                # 'goals': [(1, 1)],
                # 'initial_angles': [[0, 0]],
                'use_random_goals':True,
                'image_goal':(160,320),
                'image_only':True,
                }
    )
    gym.envs.register(
        id='NServoArmImage-v0',
        entry_point='rnr.nservoarm:NServoArmEnv',
        max_episode_steps=500,
        kwargs={#'ngoals': 10,  # 'image_goal':(160,320)
                # 'goals': [(1, 1)],
                # 'initial_angles': [[0, 0]],
                'use_random_goals':True,
                'image_goal':(160,320),
                }
    )
    gym.envs.register(
        id='NServoArm-v6',
        entry_point='rnr.nservoarm:NServoArmEnv',
        max_episode_steps=500,
        kwargs={#'ngoals': 10,  # 'image_goal':(160,320)
                'goals': [(1, 1, 0),(-1,1,pi/2),(0,1,0)],
                'links': [1,.7,.3],
                'd2reward': True,
                # 'initial_angles': [[0, 0]],
                'deadband_reward': False,
                'deadband_stop':True,
                }
    )
    gym.envs.register(
        id='Slider-v0',
        entry_point='rnr.slider:SliderEnv',
        max_episode_steps=500,
        kwargs={}
    )
    
    gym.envs.register(
        id='Pendulum-v100',
        entry_point='rnr.wrappedenvs:RestartablePendulumEnv',
        max_episode_steps=200,
    )
    
    
    gym.envs.register(
        id='ContinuousMountainCart-v100',
        entry_point='rnr.wrappedenvs:RestartableContinuousMountainCartEnv',
        max_episode_steps=200,
    )
    
    gym.envs.register(
        id='PendulumHER-v100',
        entry_point='hindsight:PendulumHEREnv',
        max_episode_steps=200,
    )

def hello_world(envname):
    env = gym.make(envname)
    for i_episode in range(200):
        observation = env.reset()
        for t in range(1000):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            print(reward,observation)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break