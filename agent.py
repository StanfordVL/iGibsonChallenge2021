from gibson2.envs.challenge import Challenge
import numpy as np

class RandomAgent:
    def reset(self):
        pass

    def act(self, observations):
        action = np.random.uniform(low=-1,high=1,size=(2,))
        return action

def main():
    agent = RandomAgent()
    challenge = Challenge()
    challenge.submit(agent)

if __name__ == "__main__":
    main()