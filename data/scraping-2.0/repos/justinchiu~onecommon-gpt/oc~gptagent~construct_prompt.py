import os
import openai
import json
import tiktoken

from oc.dot import Dot

json_file = f"{os.environ['HOME']}/research/onecommon/aaai2020/experiments/data/onecommon/final_transcripts.json"
with open(json_file, "r") as f:
    dialogues = json.load(f)

board_file = f'{os.environ["HOME"]}/research/onecommon/aaai2020/experiments/data/onecommon/shared_4.json'
with open(board_file, "r") as f:
    scenario_list = json.load(f)
BOARDS = {
    scenario['uuid']: scenario
    for scenario in scenario_list
}

OC_INSTRUCTIONS = """You are a helpful assistant.
You and your partner (They) are trying to find one dot in common.
You both see overlapping but different view of a game board.
Your view contains 7 dots, a few of which are shared with your partner.
Your goal is to discuss groups of dots in order to arrive at a single shared dot.

We give a couple example dialogues to follow below. Complete the last one."""

PROMPT_EXAMPLE_IDXS = [2,3]

def verbalize_board(board):
    # `board` should be raw json dict from shared_4.json.
    # hopefully that's what's given to process_ctx?
    dots = [Dot(dot) for dot in board]
    return "\n".join(
        f"* Dot {d.id}: Position: ({d.x}, {d.y}) Radius: {d.size} Color: {d.color}"
        for i, d in enumerate(dots)
    )

def convert_id(id, agent_id):
    return "You" if id == agent_id else "Them"

def construct_example(dialogue, agent_id):
    scenario_id = dialogue["scenario_uuid"]
    board = dialogue["scenario"]["kbs"][agent_id]

    board_desc = verbalize_board(board)

    turns = [(t["agent"], t["data"]) for t in dialogue["events"] if t["action"] == "message"]
    selects = [(t["agent"], int(t["data"].replace('"', ""))) for t in dialogue["events"] if t["action"] == "select"]

    turn_string = "\n".join([f"{convert_id(id, agent_id)}: {turn}" for id, turn in turns])
    select_string = "\n".join([f"{convert_id(id, agent_id)}: Select {turn}" for id, turn in selects])

    return "\n".join([board_desc, turn_string, select_string])


def construct_prompt(board_desc, turns, agent_id):
    examples = [
        construct_example(dialogues[d], agent_id)
        for d in [0,1]
    ]
    example_string = "\n\n".join([
        f"Example {i+1}:\n{x}"
        for i, x in enumerate(examples)
    ])
    turn_string = "\n".join(turns)
    prompt = f"""{example_string}

Example {len(examples)+1}:
{board_desc}
{turn_string}"""
    return prompt

    """
    print(prompt)
    enc = tiktoken.encoding_for_model("gpt-4")
    encoded_prompt = enc.encode(prompt)
    print(len(encoded_prompt))
    """

