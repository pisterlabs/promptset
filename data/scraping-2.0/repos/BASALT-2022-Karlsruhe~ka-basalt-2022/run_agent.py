from argparse import ArgumentParser
import pickle
import queue

import torch as th
import aicrowd_gym
import minerl

from gym.wrappers import Monitor

from openai_vpt.agent import MineRLAgent
from impala_based_models import ImpalaBinaryClassifier
from find_cave_classifier import preprocessing

ESC_MODELS = {"MineRLBasaltFindCave-v0": ImpalaBinaryClassifier}


def main(model, weights, esc_model_path, env_string, n_episodes=3, max_steps=int(1e9), show=False, record=False, video_dir="./video"):
    # Using aicrowd_gym is important! Your submission will not work otherwise
    env = aicrowd_gym.make(env_string)

    env._max_episode_steps = max_steps
    if record:
        # enable recording
        env.metadata['render.modes'] = ["rgb_array", "ansi"]
        env = Monitor(env, video_dir, video_callable=lambda episode_id: True, force=True)

    agent_parameters = pickle.load(open(model, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = MineRLAgent(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights)

    esc_model = None
    if esc_model_path is not None:
        esc_model = ESC_MODELS[env_string]()
        esc_model.load_state_dict(th.load(esc_model_path))
    esc_min_count = 40

    for _ in range(n_episodes):
        obs = env.reset()
        n_steps = 0
        esc_count = 0

        while True:
            action = agent.get_action(obs)

            if esc_model is None:
                print(f"Step {n_steps}, ESC is off.")
                action["ESC"] = 0
            else:
                img = preprocessing(obs["pov"].copy())
                img_th = th.from_numpy(img).float()
                esc_action = esc_model.predict(img_th).item()

                # Only press ESC if it is detected for x frames
                esc_probs = esc_model.class_probs(img_th)
                print(f"Step {n_steps}, ESC probs: {esc_probs}")

                esc_count += esc_action
                if esc_action == 0:
                    esc_count = 0
                if esc_count >= esc_min_count:
                    action["ESC"] = 1
                else:
                    action["ESC"] = 0

            obs, _, done, _ = env.step(action)
            if show:
                env.render()
            n_steps += 1
            if done:
                break
    env.close()


if __name__ == "__main__":
    parser = ArgumentParser("Run pretrained models on MineRL environment")

    parser.add_argument("--weights", type=str, required=True, help="Path to the '.weights' file to be loaded.")
    parser.add_argument("--model", type=str, required=True, help="Path to the '.model' file to be loaded.")
    parser.add_argument("--esc-model-path", type=str, required=True, help="Path to the '.model' file to be loaded.")
    parser.add_argument("--env-string", type=str, required=True)
    parser.add_argument("--max_steps", type=int, required=False, default=1000)
    parser.add_argument("--show", action="store_true", help="Render the environment.")
    parser.add_argument("--record", action="store_true", help="Record the rendered environment.")
    parser.add_argument("--video_dir", type=str, required=False, default="./video")

    args = parser.parse_args()

    main(args.model, args.weights, args.env, show=args.show, record=args.record, max_steps=args.max_steps, video_dir=args.video_dir)
