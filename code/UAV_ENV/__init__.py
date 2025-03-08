from gym.envs.registration import register

register(
    id='DqnUavEnv-v2',
    entry_point='UAV_ENV.envs:DQNUAVenv_SD',
)

register(
    id='DqnUavEnv-v3',
    entry_point='UAV_ENV.envs:DQNUAVenv_T1',
)
register(
    id='DqnUavEnv-v4',
    entry_point='UAV_ENV.envs:DQNUAVenv_T2',
)

register(
    id='DqnUavEnv-v5',
    entry_point='UAV_ENV.envs:DQNUAVenv_OMA',
)