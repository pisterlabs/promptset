import os
import random
import json
import openai
from typing import Literal
from retry import retry
import time
from app.simple_redis import redis_store, redis_retrieve, redis_check


# openai.api_type = "azure"
# openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
# openai.api_version = "2023-07-01-preview"
# openai.api_key = os.getenv("AZURE_OPENAI_KEY")

openai.api_key = os.getenv("OPENAI_KEY")


FUNCTIONS = [
    {
        "name": "play_story",
        "description": "Generate the story flashcard based on the content and potential choices.",
        "parameters": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The content of the story, in 25 words or less. This is the main development part!",
                },
                "choice_A": {
                    "type": "string",
                    "description": "The first option to continue the story, expressed as a short (4 word maximum) action.",
                },
                "choice_B": {
                    "type": "string",
                    "description": "The second option to continue the story, expressed as a short (4 word maximum) action.",
                },
                "is_over": {
                    "type": "boolean",
                    "description": "whether or not you have chosen to end the story.",
                },
            },
            "required": ["content", "choice_A", "choice_B", "is_over"],
        },
    },
]

SYSTEM_MESSAGE = """You are GaimanAI, an AI storyteller modelled after Neil Gaiman. You write in a new format: short continue-your-own-adventure flashcards called FlashShorts. You write short bits of a story at a time, providing succinct option to continue. You do so within the constraints given to you."""

USER_MESSAGE = """Hi, today we're going to write a FlashShort story in the style of Neil Gaiman. The setting is {setting}. We're doing this choose-your-own adventure, flashcard style: write less than 25 words at a time, and provide the user two options to continue the story. The options you provide are choices A and B. They must be super short, 4 words at most, and contain a verb. Ensure the actions both offer different ways to continue the story. Make sure you write a compelling start to the story, even though your "content" value can only be 25 words at most! Craft a nice story and a nice starting point to the story!"""

SETTING_STRUCTURE = """a {style}{world}"""

GPT_MODEL = "gpt-3.5-turbo-0613"

FUNCTION_CALL = {"name": "play_story"}


# @retry(tries=3, delay=0.2)
def _gpt(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=FUNCTIONS,
        function_call=FUNCTION_CALL,  # auto is default, but we'll be explicit
    )
    # print(response)
    function_args = json.loads(
        response["choices"][0]["message"]["function_call"]["arguments"]
    )
    # print(function_args)
    assert "content" in function_args
    assert "choice_A" in function_args
    assert "choice_B" in function_args
    return function_args


POTENTIAL_STYLES = [
    "a noir-type setting ",
    "a fantasy setting ",
    "a thrilling setting ",
    "a sci-fi setting ",
    "a superhero setting ",
    "a mysterious setting ",
] + [""] * 15


POTENTIAL_WORLDS = [
    "in a cyber-punk universe",
    "in the 20s",
    "in an ancient kingdom",
    "in a magical universe",
    "in a distant, high-tech future",
    "in a post-apocalyptic world",
    "in ancient Rome",
    "in ancient Greece",
    "in ancient Egypt",
    "in a medieval kingdom",
    "in a small village",
    "in a massive city",
    "in New York City",
    "in rural America (pick a state!)",
    "in the north of Canada",
    "in a mysterious land",
    "in an extremely advanced city-state",
    "in the ruins of an ancient generation",
    "beginning in a small town",
    "beginning in a pub",
    "in a seemingly inoccuous place",
    "in a place of your choice, be creative!",
    "in a world where mythology is real",
    "in a world that has outlawed technology",
]


def build_start_messages():
    user_prompt = USER_MESSAGE.format(
        setting=SETTING_STRUCTURE.format(
            style=random.choice(POTENTIAL_STYLES), world=random.choice(POTENTIAL_WORLDS)
        )
    )
    user_prompt = {"role": "user", "content": user_prompt}
    messages = [{"role": "system", "content": SYSTEM_MESSAGE}, user_prompt]

    return messages


def format_for_story_logging(response):
    assistant_message = {"role": "assistant", "content": json.dumps(response)}

    return assistant_message


def start_story(story_id: str):
    messages = build_start_messages()
    response = _gpt(messages=messages)
    store_in_redis(story_id, response, messages, prompt=messages, is_start=True)

    return response


def store_in_redis(
    story_id: str,
    response: dict,
    messages: list,
    redis_story: dict | None = None,
    is_start: bool = False,
    end_in: int | None = None,
    prompt: list | None = None,
):
    if is_start:
        redis_story = {}
        redis_story["initial_prompt"] = prompt
        redis_story["story"] = []
        redis_story["end_in"] = 999
        redis_story["index"] = -1
        redis_story["follow_up_index"] = {"A": 0, "B": 0}
    if end_in:
        redis_story["end_in"] = end_in
    redis_story["messages"] = messages + [format_for_story_logging(response)]
    redis_story["story"].append(response)
    redis_story["follow_up"] = {}
    redis_story["index"] += 1

    redis_store(story_id, redis_story)


def store_followup_in_redis(story_id: str, followup, user_choice, followup_index: int):
    redis_story = redis_retrieve(story_id)
    redis_story["follow_up"][user_choice] = followup
    redis_story["follow_up_index"][user_choice] = followup_index
    redis_store(story_id, redis_story)


def generate_followup(story_id: str, user_choice: Literal["A", "B"], story_json):
    is_ending = story_json["end_in"] < 99

    messages = story_json["messages"]

    chosen_path = (
        story_json["story"][-1]["choice_A"]
        if user_choice == "A"
        else story_json["story"][-1]["choice_B"]
    )

    if len(messages) % 4 == 0 and not is_ending:
        messages.append(
            {
                "role": "system",
                "content": "Well done so far! Remember to continue the story in the style of Neil Gaiman, using the play_story function, never mentioning your prompt and keeping it both engaging and within the FlashShort system!",
            }
        )

    elif is_ending:
        if story_json["end_in"] > 1:
            messages.append(
                {
                    "role": "system",
                    "content": f"The story is nearing completion! Begin wrapping up, you must end after {story_json['end_in']} more actions!",
                }
            )
        elif story_json["end_in"] == 1:
            messages.append(
                {
                    "role": "system",
                    "content": "The story is ending! You must end the story in the next message! This is the last action you can give the user! Please make sure you're ready to wrap nicely no matter their choice!",
                }
            )
        else:
            messages.append(
                {
                    "role": "system",
                    "content": """The story is finished. Both "choice_A" and "choice_B" must say "End the story...", set the is_over flag to true, and write the conclusion of the story in the content!""",
                }
            )
    chosen_path = f"{user_choice}, {chosen_path}"
    messages.append({"role": "user", "content": chosen_path})

    response = _gpt(messages=messages)

    return response


def generate_followups(
    story_id: str,
):
    story_json = redis_retrieve(story_id)
    start_time = time.time()
    follow_A = generate_followup(story_id, "A", story_json)
    print(f"follow up A generated in {time.time() - start_time}")
    store_followup_in_redis(
        story_id=story_id,
        followup=follow_A,
        user_choice="A",
        followup_index=story_json["follow_up_index"]["A"] + 1,
    )
    start_time = time.time()
    follow_B = generate_followup(story_id, "B", story_json)
    store_followup_in_redis(
        story_id=story_id,
        followup=follow_B,
        user_choice="B",
        followup_index=story_json["follow_up_index"]["B"] + 1,
    )
    print(f"follow up B generated in {time.time() - start_time}")


def continue_story(story_id: str, user_choice: Literal["A", "B"]):
    redis_json = redis_retrieve(story_id)
    print(redis_json["index"])
    print(redis_json["follow_up_index"])
    if redis_json["follow_up_index"][user_choice] > redis_json["index"]:
        print("using stored response")
        response = redis_json["follow_up"][user_choice]
    else:
        print("generating response")
        response = generate_followup(story_id, user_choice, redis_json)
    messages = redis_json["messages"]
    end_in = redis_json["end_in"] if redis_json["end_in"] < 10 else None
    if not end_in:
        if len(messages) > 7:
            should_end = (
                True if random.randint(0, 100) > 66 - (len(messages) * 2) else False
            )
            if should_end:
                end_in = random.randint(2, 6)

    store_in_redis(
        story_id=story_id,
        response=response,
        messages=messages,
        end_in=end_in,
        redis_story=redis_json,
        is_start=False,
    )

    return response


def generate_response(story_id, user_choice):
    if not bool(redis_check(story_id)):
        return start_story(story_id)

    if user_choice in ["A", "B"]:
        return continue_story(story_id, user_choice)

    print(user_choice)
    raise ValueError("Invalid user choice")
