import numpy as np

ACTION_DIM = 2
LINEAR_VEL_DIM = 0
ANGULAR_VEL_DIM = 1


class RandomAgent:
    def __init__(self):
        pass

    def reset(self):
        pass

    def act(self, observations):
        action = np.random.uniform(low=-1, high=1, size=(ACTION_DIM,))
        return action


class ForwardOnlyAgent(RandomAgent):
    def act(self, observations):
        action = np.zeros(ACTION_DIM)
        action[LINEAR_VEL_DIM] = 1.0
        action[ANGULAR_VEL_DIM] = 0.0
        return action


if __name__ == "__main__":
    obs = {
        'depth': np.ones((180, 320, 1)),
        'rgb': np.ones((180, 320, 3)),
        'sensor': np.ones((2,))
    }

    agent = RandomAgent()
    action = agent.act(obs)
    print('action', action)

    agent = ForwardOnlyAgent()
    action = agent.act(obs)
    print('action', action)
