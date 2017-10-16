import gym

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