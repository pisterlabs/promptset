import os
from argparse import ArgumentParser
import pickle

import aicrowd_gym

from openai_vpt.agent import MineRLAgent

MINERL_DATA_ROOT = os.getenv('MINERL_DATA_ROOT', 'data/')


def main(env_name, n_episodes, max_steps, show=False, counter_max=128, offset=0, power=1, warmup=0):
    """try:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.benchmark_limit = 0
    except Exception as e:
        print(e)"""
    # Using aicrowd_gym is important! Your submission will not work otherwise
    env = aicrowd_gym.make(env_name)
    # Load your model here
    # NOTE: The trained parameters must be inside "train" directory!
    in_model = os.path.join(MINERL_DATA_ROOT, "VPT-models/foundation-model-1x.model")
    in_weights = os.path.join(MINERL_DATA_ROOT, "VPT-models/foundation-model-1x.weights")

    agent_parameters = pickle.load(open(in_model, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    model = MineRLAgent(policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs, device="cuda",
                        do_custom=True, set_location=f"./train/states2actions_{env_name}.npz", counter_max=counter_max, offset=offset)
    model.load_weights(in_weights)
    for i in range(n_episodes):
        obs = env.reset()
        model.reset()
        done = False
        if show:
            import cv2
            video = cv2.VideoWriter(f'{env_name}_episode_{i}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), float(20), (256, 256))
        for step_counter in range(max_steps):
            action, inv = model.get_action_custom(obs, step_counter < warmup, power=power)
            if step_counter < max_steps and step_counter < 2000:
                if not inv:
                    action["ESC"] = 0

            obs, reward, done, info = env.step(action)
            if show:
                video.write(cv2.cvtColor(cv2.resize(env.render(mode="human"), (256, 256), cv2.INTER_AREA), cv2.COLOR_BGR2RGB))
            if done:
                break
        if show:
            video.release()

        print(f"[{i}] Episode complete")

    # Close environment and clean up any bigger memory hogs.
    # Otherwise, you might start running into memory issues
    # on the evaluation server.
    env.close()


if __name__ == "__main__":
    parser = ArgumentParser("Run pretrained models on MineRL environment")

    parser.add_argument("--weights", type=str, required=True, help="Path to the '.weights' file to be loaded.")
    parser.add_argument("--model", type=str, required=True, help="Path to the '.model' file to be loaded.")
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--show", action="store_true", help="Render the environment.")

    args = parser.parse_args()

    main(args.model, args.weights, args.env, show=args.show)
