import os
from argparse import ArgumentParser
import pickle

import aicrowd_gym

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from openai_vpt.agent import MineRLAgent

# from cliff_detector import AnimalDetector
from waterfall.waterfall import makeWaterFall, near_magma, turn, is_stuck, is_in_water, get_cur_obs, do_none_ac, \
    forwardjump, back
from utils.tools import unzip, BubbleCheck


class ToWardsState:
    def __init__(self) -> None:
        self.YMAX = 180
        self.XMAX = 360

        self.x = 90.0
        self.y = 90.0

    def mouse_move(self, dx, dy):
        self.x = (self.x + dx) % self.XMAX

        self.y = min(max(0, (self.y + dy)), self.YMAX)

    def reset(self):
        self.x = 90.0
        self.y = 90.0

    def __repr__(self) -> str:
        return f"x: {self.x}, y: {self.y}"

def main(model, weights, env, n_episodes=3, max_steps=int(6000), show=False):
    # Using aicrowd_gym is important! Your submission will not work otherwise
    env = aicrowd_gym.make(env)
    agent_parameters = pickle.load(open(model, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = MineRLAgent(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs, use_legal_action=True)
    agent.load_weights(weights)

    # cliff_detector = AnimalDetector('../yolov5/runs/train/exp6/weights/last.pt', 'P2.v2i.yolov5pytorch/data.yaml')
    base_dir = os.path.dirname(os.path.dirname(__file__))
    # if not os.path.exists(os.path.join(base_dir, 'train', 'resnet_500_best.pb')) or \
    #         not os.path.exists(os.path.join(base_dir, 'hub/checkpoints/resnet50-0676ba61.pth')):
    #     unzip(base_dir)

    from waterfall.classify import CliffDetector
    cliff_detector = CliffDetector(os.path.join(base_dir, 'train', 'resnet_500_best.pb'))
    bubble_checker = BubbleCheck(template_path=os.path.join(base_dir, 'utils/bubble_template.png'),
                                 mask_path=os.path.join(base_dir, 'utils/bubble_mask.png'))

    for _ in range(n_episodes):
        towardstate = ToWardsState()
        obs = env.reset()

        last_10_obs = [None for _ in range(10)]
        # makeWaterFall(env, towardstate)
        # continue

        i = 0
        while i < max_steps:
        # for i in range(max_steps):
            action = agent.get_action(obs)
            # ESC is not part of the predictions model.
            # For baselines, we just set it to zero.
            # We leave proper execution as an exercise for the participants :)
            action["ESC"] = 0
            obs, _, done, _ = env.step(action)

            dy, dx = action['camera'][0]
            towardstate.mouse_move(dx, dy)
            last_10_obs = last_10_obs[1:] + [obs['pov']]

            try:
                for _ in range(17):
                    if near_magma(obs['pov']):
                        obs = turn(env, towardstate, [towardstate.x - 20, towardstate.y], min_diff=9)
                        i += 2
                    else:
                        break

                if bubble_checker.check(obs['pov']):
                    obs = forwardjump(env)
                    i += 10
                elif i % 100 == 0:
                    if i > 10:
                        is_stuck(env, towardstate, last_10_obs)

                terminal = cliff_detector.check(obs['pov']) and (not is_in_water(obs['pov']))
                if terminal:
                    print(f"start second check at step {i}...")
                    do_none_ac(env, 6)
                    back(env, 1)
                    i += 12
                    for turn_x in [1, 1, 1, -4, -1, -1]:
                        obs = turn(env, towardstate, [towardstate.x + turn_x, towardstate.y])
                        i += 1
                        terminal = cliff_detector.check(obs['pov'])
                        if terminal: break

                if terminal:
                    print(f"make waterfall in step {i}")
                    makeWaterFall(env, towardstate, high=3)
                if i > 5000:
                    makeWaterFall(env, towardstate, high=15)
                    print(f"make waterfall in step {i}")
                    done = True
            except Exception as e:
                print("some error occurred")
                done = True

            i += 1
            if show:
                env.render()
            if done or terminal:
                print("reset env...")
                break
    env.close()


if __name__ == "__main__":
    parser = ArgumentParser("Run pretrained models on MineRL environment")

    parser.add_argument("--weights", type=str, required=True, help="Path to the '.weights' file to be loaded.")
    parser.add_argument("--model", type=str, required=True, help="Path to the '.model' file to be loaded.")
    parser.add_argument("--env", type=str, default='MineRLBasaltMakeWaterfall-v0')
    parser.add_argument("--show", action="store_true", help="Render the environment.")

    args = parser.parse_args()

    main(args.model, args.weights, args.env, show=args.show, n_episodes=10)
