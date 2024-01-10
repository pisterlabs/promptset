import os
from argparse import ArgumentParser
import pickle

import aicrowd_gym
import minerl

from openai_vpt.agent import MineRLAgent

from cliff_detector import AnimalDetector


def main(model, weights, env, n_episodes=3, max_steps=int(6000), show=False, human=False):
    # Using aicrowd_gym is important! Your submission will not work otherwise
    env = aicrowd_gym.make(env)
    if human:
        from minerl.human_play_interface.human_play_interface import HumanPlayInterface
        env = HumanPlayInterface(env)

    agent_parameters = pickle.load(open(model, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = MineRLAgent(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs, use_legal_action=True)
    agent.load_weights(weights)

    cliff_detector = AnimalDetector('../yolov5/runs/train/P3.v3i.1200/weights/best.pt', 'P2.v2i.yolov5pytorch/data.yaml')
    for _ in range(n_episodes):
        obs = env.reset()

        for i in range(max_steps):
            if human:
                action = None
            else:
                action = agent.get_action(obs)
                # ESC is not part of the predictions model.
                # For baselines, we just set it to zero.
                # We leave proper execution as an exercise for the participants :)
                action["ESC"] = 0
            obs, _, done, _ = env.step(action)

            terminal = cliff_detector.check(obs['pov'])

            if show:
                env.render()
            if done or terminal and not human:
                print("reset env...")
                break
    env.close()


if __name__ == "__main__":
    parser = ArgumentParser("Run pretrained models on MineRL environment")

    parser.add_argument("--weights", type=str, required=True, help="Path to the '.weights' file to be loaded.")
    parser.add_argument("--model", type=str, required=True, help="Path to the '.model' file to be loaded.")
    parser.add_argument("--env", type=str, default='MineRLBasaltMakeWaterfall-v0')
    parser.add_argument("--show", action="store_true", help="Render the environment.")
    parser.add_argument("--human", type=bool, default=True, help="Render the environment.")

    args = parser.parse_args()

    main(args.model, args.weights, args.env, show=args.show, n_episodes=10, human=args.human, max_steps=10000000000000000)
