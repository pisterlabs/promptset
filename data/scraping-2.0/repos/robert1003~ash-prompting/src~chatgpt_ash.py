# Same code as notebook but as a script to run easily
# Code to run chat based ASH and ReAct on chatgpt (see use_ash option below to change to ReAct or ASH)
import argparse
import random
import string
import openai
from dotenv import load_dotenv

load_dotenv()

from retry import retry
import re
import os
import sys

sys.path.insert(0, "../WebShop")
from web_agent_site.envs.web_agent_text_env import WebAgentTextEnv

openai.api_key = "<INSERT API KEY HERE>"


# Setup environment
env = WebAgentTextEnv(
    observation_mode="text_rich",
    render=False,
    num_products=None,  # 1000 for small product space, None for full product space
)

actor_num_examples = 2


@retry(tries=2, delay=20)
def chatgpt(prior_prompt, cur_traj, type="actor", max_tokens=100, stop=["\n\n"]):
    if type == "actor":
        messages = [
            {
                "role": "system",
                "content": prior_prompt["intro"]
                + (
                    "\nYou are given few solved examples to help you understand your task.\n"
                    if actor_num_examples > 0
                    else ""
                ),
            },
        ]
        for ex_num, example in enumerate(prior_prompt["examples"]):
            i = 1
            for interaction in example:
                messages += [
                    {
                        "role": "system",
                        "name": "example_user",
                        "content": (
                            f"### Instruction:\n{example[0]['ob']}\n\n"
                            if i == 1
                            else ""
                        )
                        + f"### Observation {i}:\n"
                        + interaction["ob"],
                    },
                    {
                        "role": "system",
                        "name": "example_assistant",
                        "content": f"### Action {i}:\n" + interaction["act"],
                    },
                ]
                i += 1

        i = 1
        for interaction in cur_traj[1:-1]:
            messages += [
                {
                    "role": "user",
                    "content": (
                        f"### Instruction:\n{cur_traj[0]['ob']}\n\n" if i == 1 else ""
                    )
                    + f"### Observation {i}:\n"
                    + interaction["ob"],
                },
                {
                    "role": "assistant",
                    "content": f"### Action {i}:\n{interaction['act']}",
                },
            ]
            i += 1

        messages += [
            {
                "role": "user",
                "content": (
                    f"### Instruction:\n{cur_traj[0]['ob']}\n\n"
                    if len(cur_traj) <= 2
                    else ""
                )
                + f"### Observation {i}:\n"
                + cur_traj[-1]["ob"],
            }
        ]

    elif type == "summarizer":
        messages = [
            {"role": "system", "content": prior_prompt["intro"]},
        ]
        for ex_num, example in enumerate(prior_prompt["examples"]):
            messages += [
                {
                    "role": "system",
                    "name": "example_user",
                    "content": f"### Instruction:\n{example['instr']}\n\n### Previous Action of Agent:\n{example['act']}\n\n### Observation:\n{example['ob']}",
                },
                {
                    "role": "system",
                    "name": "example_assistant",
                    "content": f"### Condensed Observation:\n" + example["resp"],
                },
            ]
        messages += [
            {
                "role": "user",
                "content": f"### Instruction:\n{cur_traj[0]['ob']}\n\n### Previous Action of Agent:\n{cur_traj[-2]['act'] if cur_traj[-2]['act'] != '' else 'None'}\n\n### Observation:\n{cur_traj[-1]['ob']}",
            },
        ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop,
    )
    return response["choices"][0]["message"]["content"]


def ob2summary(prior_prompt, cur_traj, llm):
    try:
        summary = llm(
            prior_prompt, cur_traj, type="summarizer", max_tokens=500, stop=["\n\n"]
        ).strip()

        template = r"info\[[\S\s]+\]"
        for res in re.finditer(template, summary):
            info = res.group()
            if info.startswith("info["):
                info = info[5:]
            if info.endswith("]"):
                info = info[:-1]

            if "[Search]" in info:
                idx = info.find("[Search]")
                info = info[: idx + len("[Search]")]

            if "[Buy Now]" in info:
                idx = info.find("[Buy Now]")
                info = info[: idx + len("[Buy Now]")]

            return info, None, summary

        return None, f"no summary found in {summary}", summary

    except Exception as e:
        return None, e, None


def summary2act(prior_prompt, cur_traj, llm, env_info):
    try:
        act = llm(prior_prompt, cur_traj, type="actor", stop=["\n\n"]).strip()

        available_acts = ["search", "click", "think"]
        template = "(" + "|".join(available_acts) + r")\[[^\[\]]+\]"
        for res in re.finditer(template, act):
            act_str = res.group().strip()
            if act_str.startswith("think"):
                return act_str, None, act
            elif act_str.startswith("search"):
                if env_info["has_search_bar"]:
                    return act_str, None, act
            elif act_str.startswith("click"):
                if act_str.lower() in list(
                    map(lambda x: f"click[{x}]", env_info["clickables"])
                ):
                    return act_str, None, act

        return None, f"no act found in {act}", act

    except Exception as e:
        return None, e, None


def process_instruction(instr):
    template = r"(?<=Instruction:)([\S\s]+)(?=\n)"
    res = re.match(template, instr)
    if res is None:
        return instr.strip()

    return res.group(0).strip()


# Get actor, summary prompts for chatgpt
import sys

sys.path.append("../prompts")
from chatgpt_prompts import actor_prompt, summarizer_prompt
from tqdm import tqdm

resume_idx = 184
llm = chatgpt
max_fail = 5
max_iter = 20
seed = 501
runs = 500
use_ash = True  # Make this False if you need ReAct
retain_latest_k = -1

if retain_latest_k > 0:
    actor_prompt["intro"] = actor_prompt["intro"].format(
        "Only the last {} Observation - Action cycle(s) is shown to you.".format(
            retain_latest_k
        )
    )
else:
    actor_prompt["intro"] = actor_prompt["intro"].format("")

# Change filename accordingly here
log_file = open("./log_ash_2_ex_full", "a+")

for i in tqdm(range(resume_idx, runs)):
    done = False
    counter = 0
    invalid_counter = 0
    traj = []
    rew = 0
    need_summary = True
    term_status = None
    act = ""

    # print(f'Instance {i}: seed {seed+i}')
    print(f"Instance {i}: seed {seed+i}", file=log_file, flush=True)
    random.seed(seed + i)
    session = "".join(random.choices(string.ascii_lowercase, k=10))
    (ob, _) = env.reset(session=session)
    info = env.get_available_actions()
    # Instruction with blank action added as the first interaction
    ob = "\n".join(ob.strip().split("\n\n"))
    traj.append({"ob": process_instruction(ob), "act": ""})
    print(f'### Instruction:\n{traj[0]["ob"]}\n', file=log_file, flush=True)

    while True:
        # terminate if max_iter reached
        counter += 1

        # Observation
        traj.append({"ob": ob, "act": ""})

        # print(f'### Observation {counter}:\n{ob}\n')
        print(f"### Observation {counter+1}:\n{ob}\n", file=log_file, flush=True)

        # Get summary of observation, if needed
        if use_ash and need_summary:
            summary, err, summary_with_reason = ob2summary(summarizer_prompt, traj, llm)
            if err is not None:
                # raise Exception(f'Error in ob2summary() of trajectory {seed}:', err)
                rew = 0
                term_status = "summarizer_error"
                break

            if ob.endswith("\nThought Through."):
                summary += "\nThought Through."
            if ob.endswith("\nInvalid Action."):
                summary += "\nInvalid Action."
            # print(f'### Summary {counter}:\n{summary}\n')
            print(f"### Summary {counter+1}:\n{summary}\n", file=log_file, flush=True)
            traj[-1]["ob"] = summary
            ob = summary

        # print(traj)
        # Get action
        act, err, act_full = summary2act(actor_prompt, traj, llm, info)
        if err is not None and (not isinstance(err, str) or "no act found" not in err):
            # raise Exception(f'Error in summary2act() of trajectory {seed}:', err)
            rew = 0
            term_status = "max_token_limit"
            break

        if ob.endswith("\nThought Through."):
            ob = ob[: -len("\nThought Through.")]
        if ob.endswith("\nInvalid Action."):
            ob = ob[: -len("\nInvalid Action.")]

        # Case by case analysis for action
        if err is not None and "no act found" in err:
            act = act_full
            ob = ob + "\nInvalid Action."
            need_summary = True
            invalid_counter += 1
            if invalid_counter >= max_fail:
                rew = 0
                term_status = "max_fail"
                break

            # print(f'### Action {counter+1}: {act} (invalid)\nn')
            print(
                f"### Action {counter+1}:\n{act} (invalid)\n", file=log_file, flush=True
            )
        else:
            invalid_counter = 0
            if act.startswith("think"):
                ob = ob + "\nThought Through."
                need_summary = True
            else:
                ob, rew, done, _ = env.step(act)
                ob = "\n".join(ob.strip().split("\n\n"))
                info = env.get_available_actions()
                need_summary = True

            # print(f'### Action {counter+1}: {act}\n')
            print(f"### Action {counter+1}:\n{act}\n", file=log_file, flush=True)

            if done:
                break

        traj[-1]["act"] = act

        if counter >= max_iter:
            rew = 0
            term_status = "max_iter"
            break

    # print('reward', rew, 'term_status', term_status)
    print(
        "reward",
        rew,
        "term_status",
        term_status,
        "\n===============================\n\n",
        file=log_file,
        flush=True,
    )
