from gym.envs.registration import register

# Env registration
# ==========================

register(
    id='TicTacToe-v1',
    entry_point='gym_basicgame.basicgame:BasicGameEnv'
)