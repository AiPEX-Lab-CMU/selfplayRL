from gym.envs.registration import register

# Env registration
# ==========================

register(
    id='BasicGame-v1',
    entry_point='gym_basicgame.basicgame:BasicGameEnv'
)