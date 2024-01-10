
import os
import glob
import json
import logging
import argparse
import datetime

from logging import INFO
from os.path import join as pjoin

import openai
import tiktoken
from tqdm import tqdm
from termcolor import colored
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

EXAMPLE_FILE = pjoin(os.path.dirname(__file__), "example.txt")

def clean(s):
    clean_toks = ['\n', '\t']
    for tok in clean_toks:
        s = s.replace(tok, '. ')

    # Remove double points
    while '. .' in s:
        s = s.replace('. .', '. ')

    return s

@retry(
    reraise=True,
    stop=stop_after_attempt(1000),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=(
        retry_if_exception_type(openai.error.Timeout)
        | retry_if_exception_type(openai.error.APIError)
        | retry_if_exception_type(openai.error.APIConnectionError)
        | retry_if_exception_type(openai.error.RateLimitError)
    ),
)
def llm_gpt(prompt, model="gpt-3.5-turbo", pbar=None, **kwargs):
    try:
        if model == "gpt-4-0613":
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                api_key=os.getenv("OPENAI_API_KEY2"),
                api_base="https://api.openai.com/v1",
                api_type="open_ai",
                api_version="2020-11-07",
                **kwargs,
            )
        elif openai.api_type == "azure":
            # When using the Azure API, we need to use engine instead of model argument.
            response = openai.ChatCompletion.create(
                engine=model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
        else:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
    except Exception as e:
        if pbar:
            pbar.set_postfix_str(f"[{datetime.datetime.now()}] {e}")
        raise e

    choice = response["choices"][0]
    output = choice["message"]["content"].strip()
    return output


def check_winnability(gamefile, model_name, random_seed, env_step_limit, logger=None):
    logger = logger or logging.getLogger()

    # Import environment
    #sys.path.append(os.path.dirname(gamefile))
    TextGame = __import__(os.path.basename(gamefile[:-3])).TextGame

    # Load ICL example
    with open(EXAMPLE_FILE) as f:
        example = f.read()

    # Load encoding tool to count token numbers
    encoding = tiktoken.encoding_for_model(model_name)

    # Initialize environment
    env = TextGame(randomSeed=random_seed)
    task_description = env.getTaskDescription()
    possible_actions = env.generatePossibleActions()
    recent_actions = []

    obs = env.observationStr if hasattr(env, "observationStr") else ""

    done = False
    score = 0.0
    step = 0
    game_won = False
    action = ""

    # Because we are using ReAct prompt, we allow for twice the amount of steps.
    max_steps = env_step_limit * 2

    # Given a list of string actions, compress those that are similar.
    actions = list(possible_actions.keys())
    compressed_actions = [" ".join('<NUM>' if w.isdigit() else w for w in a.split()) for a in actions]

    # If we are able to remove more than 30 actions, compression was succesful.
    if len(set(actions)) - len(set(compressed_actions)) > 30:
        actions = sorted(set(compressed_actions))

    init_prompt = 'You are playing a text-based games. Interact with the environment to solve a task.\n'
    init_prompt += "Here is an example.\n"
    init_prompt += example
    init_prompt += f"\nThe game you are about to play only understands one command at a time from the following list of commands: {actions}.\n"
    init_prompt += "Prepend your thoughts with 'think:' when planning your next steps.\n"
    init_prompt += "When you think the task is completed, say 'done'.\n"
    init_prompt += "If you think the task can't be completed at all, say 'bug'.\n"

    prompt = '\n\nHere is the task:\n' + clean(obs) + '\n' + task_description + '\n>'

    logger.info("Prompt: " + colored(init_prompt, "cyan") + colored(prompt, "yellow"))

    # Different models have different maximun token numbers
    if model_name == "gpt-3.5-turbo":
        max_len = 4096
    elif model_name == "gpt-4":
        max_len = 8192
    else:
        max_len = 4097

    pbar = tqdm(total=max_steps, desc="Steps", unit="step")
    while not done:
        pbar.update(1)

        # Cut the prompt to make it shorter than maximun token numbers
        while len(encoding.encode(init_prompt + prompt)) > max_len - 60:
            index1 = init_prompt.find('>')

            # If init prompt doesn't have actions, cut game prompt
            if index1 == -1:
                index1_prompt = prompt.find('>')
                index2_prompt = prompt.find('>', index1_prompt+1)
                prompt = prompt[:index1_prompt] + prompt[index2_prompt:]

            # Cut initial prompt
            else:
                index2 = init_prompt.find('>', index1+1)
                if index2 == -1:
                    init_prompt = init_prompt[:index1]
                else:
                    init_prompt = init_prompt[:index1] + init_prompt[index2:]

        action = llm_gpt(init_prompt + prompt, stop=['\n'], model=model_name, pbar=pbar).strip("> ")
        pbar.set_postfix_str("")
        action = action.strip()
        recent_actions.append(action)

        # Don't need to actually do think/bug/done actions.
        if action == 'bug':
            prompt += f' {action}\n'
            logger.info(colored(f' {action}', 'green'))
            break
        elif action == 'done':
            prompt += f' {action}\n'
            logger.info(colored(f' {action}', 'green'))
            break
        elif action.startswith('think:'):
            obs = 'OK.'
        else:
            obs, score, reward, done, game_won = env.step(action)

            # Given a list of string actions, compress those that are similar.
            actions = list(possible_actions.keys())
            compressed_actions = [" ".join('<NUM>' if w.isdigit() else w for w in a.split()) for a in actions]

            # If we are able to remove more than 30 actions, compression was succesful.
            if len(set(actions)) - len(set(compressed_actions)) > 30:
                actions = sorted(set(compressed_actions))

        obs = clean(obs)

        if obs == "I don't understand that.":
            obs += f" Think about why the last command was incorrect and find a solution.\n"
            obs += f" The game only understands commands from the following list: {actions}."

        # Add action and observaton to game prompt
        logger.info(colored(f' {action}', 'green') + f'\n{obs}')
        prompt += f' {action}\n{obs}\n>'

        step += 1
        if (step >= max_steps) or done or game_won:
            break

    stats = {}
    stats["gpt_done"] = (action == 'done')
    stats["gpt_bug"] = (action == 'bug')
    stats["num_actions"] = len(actions)
    stats["score"] = score
    stats["game_won"] = game_won
    stats["done"] = done
    stats["step"] = step
    stats["max_steps"] = max_steps
    stats["history"] = recent_actions
    stats["transcript"] = prompt
    stats["init_prompt"] = init_prompt

    logger.info("Run completed...")

    return stats


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game_folder", default="../cleaned_generated_game", help="Path to a folder containing BYTESIZED32 games.")
    parser.add_argument("--max_reflection_steps", type=int, default=3)
    parser.add_argument("--env_step_limit", type=int, default=30)
    parser.add_argument("--random-seed", type=int, default=20230614)
    parser.add_argument("--output_path", default="./agent_output/")
    parser.add_argument("--no_stop", action="store_true", default=False)
    parser.add_argument("--prompt_file", default="ReAct_baseline/prompt.jsonl")
    parser.add_argument("--model_name", default="gpt-3.5-turbo")
    parser.add_argument("--force", action="store_true")

    return parser.parse_args()


def init_logger(args, gamefile, log_level=INFO):
    logger = logging.getLogger()
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s][%(levelname)s\t] %(message)s",
                                    datefmt='%Y-%m-%d %H:%M:%S')
    logger.setLevel(log_level)

    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    filename = pjoin(args.output_path, os.path.basename(gamefile)[:-3] + ".log")
    fh = logging.FileHandler(filename)
    fh.setLevel(log_level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def main():
    args = parse_args()
    print(args)

    os.makedirs(args.output_path, exist_ok=True)

    stats = {}
    stats_json = pjoin(args.output_path, "eval_gpt_agent_20230621_102500.json")
    if os.path.isfile(stats_json):
        with open(stats_json) as f:
            stats = json.load(f)

    gamefiles = sorted(glob.glob(pjoin(args.game_folder, "*.py")))
    pbar = tqdm(gamefiles)
    for gamefile in pbar:
        if "_reflection_" in gamefile:
            continue

        game_file_name = os.path.basename(gamefile)
        if game_file_name in stats and not args.force:
            continue

        pbar.set_description(game_file_name)

        # use the last reflection results
        for i in range(args.max_reflection_steps)[::-1]:
            if os.path.exists(pjoin(args.game_folder, f"{game_file_name[:-3]}_reflection_{i}.py")):
                game_file_name = f"{game_file_name[:-3]}_reflection_{i}.py"

        gamefile = pjoin(args.game_folder, game_file_name)

        logger = init_logger(args, gamefile)
        logger.info(args)
        try:
            stats[game_file_name] = eval_gamefile(gamefile, args.model_name, args.random_seed, args.env_step_limit, logger)
        except Exception as e:
            stats[game_file_name] = str(e)

        with open(stats_json, 'w') as f:
            json.dump(stats, f, indent=4)


if __name__ == "__main__":
    main()
