from argparse import ArgumentParser
import pickle

import aicrowd_gym
import minerl
import cv2

from openai_vpt.agent import MineRLAgent

def main(model, weights, env, n_episodes=3, max_steps=3600, show=False, save=False):
    # Using aicrowd_gym is important! Your submission will not work otherwise
    env = aicrowd_gym.make(env)
    agent_parameters = pickle.load(open(model, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = MineRLAgent(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights)

    VIDEO_FPS = 20
    VIDEO_SHAPE = (400, 300)
    video_writer = cv2.VideoWriter(
        "agent3.mp4", cv2.VideoWriter_fourcc(*"mp4v"), VIDEO_FPS, VIDEO_SHAPE
    )

    for _ in range(n_episodes):
        obs = env.reset()
        for _ in range(max_steps):
            action = agent.get_action(obs)
            # ESC is not part of the predictions model.
            # For baselines, we just set it to zero.
            # We leave proper execution as an exercise for the participants :)
            action["ESC"] = 0
            obs, _, done, _ = env.step(action)
            if show:
                env.render()
            if save:
                frame = env.render("rgb_array")

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, VIDEO_SHAPE)

                video_writer.write(frame)

            if done:
                break
    env.close()

    video_writer.release()
    del video_writer


if __name__ == "__main__":
    parser = ArgumentParser("Run pretrained models on MineRL environment")

    parser.add_argument("--weights", type=str, required=True, help="Path to the '.weights' file to be loaded.")
    parser.add_argument("--model", type=str, required=True, help="Path to the '.model' file to be loaded.")
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--show", action="store_true", help="Render the environment.")
    parser.add_argument("--save", action="store_true", help="Save the frames as a video.")

    args = parser.parse_args()

    main(args.model, args.weights, args.env, show=args.show, save=args.save)
