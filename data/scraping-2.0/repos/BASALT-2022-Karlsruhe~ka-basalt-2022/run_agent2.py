from argparse import ArgumentParser
import pickle

import aicrowd_gym
import minerl

from gym.wrappers import Monitor

from openai_vpt.agent import MineRLAgent


def main(
    model,
    weights,
    env,
    n_episodes=3,
    max_steps=int(1e9),
    show=False,
    record=False,
    video_dir="./video",
):
    # Using aicrowd_gym is important! Your submission will not work otherwise
    env = aicrowd_gym.make(env)

    env._max_episode_steps = max_steps
    if record:
        # enable recording
        env.metadata["render.modes"] = ["rgb_array", "ansi"]
        env = Monitor(
            env, video_dir, video_callable=lambda episode_id: True, force=True
        )

    agent_parameters = pickle.load(open(model, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = MineRLAgent(
        env, device="cpu", policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs
    )
    agent.load_weights(weights)

    for _ in range(n_episodes):
        obs = env.reset()
        while True:
            action = agent.get_action(obs)
            # ESC is not part of the predictions model.
            # For baselines, we just set it to zero.
            # We leave proper execution as an exercise for the participants :)
            action["ESC"] = 0
            obs, _, done, _ = env.step(action)
            if show:
                env.render()
            if done:
                break
    env.close()


if __name__ == "__main__":
    parser = ArgumentParser("Run pretrained models on MineRL environment")

    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to the '.weights' file to be loaded.",
        default="data/VPT-models/foundation-model-1x.weights",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the '.model' file to be loaded.",
        default="data/VPT-models/foundation-model-1x.model",
    )
    parser.add_argument(
        "--env", type=str, required=True, default="MineRLBasaltFindCave-v0"
    )
    parser.add_argument("--max_steps", type=int, required=False, default=1000)
    parser.add_argument("--show", action="store_true", help="Render the environment.")
    parser.add_argument(
        "--record", action="store_true", help="Record the rendered environment."
    )
    parser.add_argument("--video_dir", type=str, required=False, default="./video")

    args = parser.parse_args()

    show = True

    main(
        args.model,
        args.weights,
        args.env,
        show=args.show or show,
        record=args.record,
        max_steps=args.max_steps,
        video_dir=args.video_dir,
    )
