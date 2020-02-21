import argparse
from IPython import embed

from simple_agent import RandomAgent, ForwardOnlyAgent
from rl_agent import SACAgent

from gibson2.envs.challenge import Challenge


def get_agent(agent_class, ckpt_path, success_distance=0.36):
    if agent_class == "Random":
        return RandomAgent(success_distance)
    elif agent_class == "ForwardOnly":
        return ForwardOnlyAgent(success_distance)
    elif agent_class == "SAC":
        return SACAgent(root_dir=ckpt_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-class", type=str, default="Random", choices=["Random", "ForwardOnly", "SAC"])
    parser.add_argument("--ckpt-path", default="", type=str)
    args = parser.parse_args()

    agent = get_agent(
        agent_class=args.agent_class,
        ckpt_path=args.ckpt_path
    )
    challenge = Challenge()
    challenge.submit(agent)


if __name__ == "__main__":
    main()
