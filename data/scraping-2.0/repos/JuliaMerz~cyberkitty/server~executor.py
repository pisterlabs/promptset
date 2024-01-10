import backoff
import tiktoken
import openai
from collections.abc import Generator
from typing import cast, Optional
from openai import OpenAI
from sqlmodel.orm.session import Session
from .models import User, Query, ApiCall, Message, LinkableObject, StoryOutline, Story, SceneOutline, ChapterOutline, Scene
from .utils import calc_cost
from .config import get_settings

conf = get_settings()

client = OpenAI(api_key=conf.OPENAI_API_KEY, base_url=conf.OPENAI_BASE_URL)


CONTINUE_PROMPT = "Your last message got cutoff, without repeating yourself, please continue writing exactly where you left off."

MAX_RETRIES = conf.MAX_RETRIES
QUERY_MAX_TOKENS = conf.QUERY_MAX_TOKENS
QUERY_TEMPERATURE = conf.QUERY_TEMPERATURE
QUERY_FREQUENCY_PENALTY = conf.QUERY_FREQUENCY_PENALTY


@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APITimeoutError))
def completions_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)

def pair_query_with_object(query: Query, obj: LinkableObject):
    if obj.__class__.__name__ == "Story":
        query.story_id = cast(int, obj.id)
    elif obj.__class__.__name__ == "StoryOutline":
        query.story_outline_id = cast(int, obj.id)
        query.story_id = cast(StoryOutline, obj).story_id
        print("SETTING QUERY STORY ID TO: ", query.story_id)
    elif obj.__class__.__name__ == "ChapterOutline":
        query.story_id = cast(ChapterOutline, obj).story_outline.story_id
        print("SETTING QUERY STORY ID TO: ", query.story_id)
        query.chapter_outline_id = cast(int, obj.id)
    elif obj.__class__.__name__ == "SceneOutline":
        query.scene_outline_id = cast(int, obj.id)
        query.story_id = cast(SceneOutline, obj).chapter_outline.story_outline.story_id
        print("SETTING QUERY STORY ID TO: ", query.story_id)
    elif obj.__class__.__name__ == "Scene":
        query.scene_id = cast(int, obj.id)
        query.story_id = cast(Scene, obj).scene_outline.chapter_outline.story_outline.story_id
        print("SETTING QUERY STORY ID TO: ", query.story_id)

    # we need to store the story either way.
    # this is a limitation in sqlalchemy—we'd want to use query.root_story_id instead



def query_executor(db: Session, system_prompt: str, prompt: str, user: User, obj: Optional[LinkableObject]=None, previous_messages: list[Message] = []) -> Generator[str | Query, None, None]:
    """
    Execute a query against the openai API.

    Side Effects:
    Generates a Query objects and associated ApiCall objects and saves them to the database.
    """

    if conf.SYS_PROMPT_PREFIX:
        system_prompt = conf.SYS_PROMPT_PREFIX + system_prompt

    messages = [
        {"role": "system", "content": system_prompt},
    ]
    for message in previous_messages:
        messages.append({"role": message.role, "content": message.content})
    messages.append({"role": "user", "content": prompt})

    complete_output = ""
    retry_count = 0

    query = Query(author_id=cast(int, user.id), original_prompt=prompt, system_prompt=system_prompt,
                   complete_output=complete_output)
    query.previous_messages=previous_messages
    if obj is not None:
        print("pairing")
        pair_query_with_object(query, obj)
    db.add(query)
    db.commit()
    db.refresh(query)
    print("ATTEMPTING QUERY: ", system_prompt, prompt)

    # $0.01 / 1K tokens	$0.03 / 1K tokens

    while True:
        try:
            stream = completions_with_backoff(model="gpt-4-1106-preview",
                                                messages=messages,
                                                max_tokens=QUERY_MAX_TOKENS,
                                                temperature=QUERY_TEMPERATURE,
                                                frequency_penalty=QUERY_FREQUENCY_PENALTY,
                                                stream=True)


            response_text = ""
            finish_reason = None
            for chunk in stream:
                choice = chunk.choices[0]
                delta =  choice.delta
                if choice.finish_reason:
                    finish_reason = choice.finish_reason

                if delta.content:
                    response_text += delta.content
                    yield delta.content

            print("RECEIVED: ", response_text)

            call_cost = None
            # if response.usage:
            #     call_cost = calc_cost(
            #         response.usage.prompt_tokens, response.usage.completion_tokens)
            #     tiktoken_call_cost = calc_cost(
            #         num_tokens_from_messages(messages), num_tokens_from_messages([{"role": "assistant", "content": choice.message.content}]))

            #     print("COST COMPARISON (should be even, api cost first): ", call_cost, tiktoken_call_cost)
            tiktoken_call_cost = calc_cost(
                num_tokens_from_messages(messages), num_tokens_from_messages([{"role": "assistant", "content": response_text}]))
            call_cost = tiktoken_call_cost

            api_call = ApiCall(query_id=cast(int, query.id), success=True, cost=call_cost,
                                output=response_text)
            api_call.input_messages=[Message(**x) for x in messages]
            db.add(api_call)
            db.commit()

            if response_text:
                complete_output += response_text
                query.complete_output = complete_output

            if finish_reason == None:
                print("WARNING: NO FINISH REASON DETECTED, LIKELY ISSUE")
            # CONTINUE CASE
            if finish_reason == "length":
                new_message = {"role": "assistant", "content": response_text}
                messages.append(new_message)
                messages.append({"role": "user", "content": CONTINUE_PROMPT})

                retry_count = 0
                continue
            # SUCCESS CASE
            if finish_reason == "stop":
                new_message = {"role": "assistant", "content": response_text}
                messages.append(new_message)
                break

            # ERROR CASES
            elif finish_reason == "content_filter":
                print("Content Filtering ERROR: " + finish_reason)
                api_call.success = False
                api_call.error = "content_filter"

                retry_count += 1

                if retry_count > MAX_RETRIES:
                    break
                continue
            else:
                # other error case
                print("Unknown ERROR: " + finish_reason if finish_reason else "NONE")
                api_call.success = False
                api_call.error = "unknown"

                retry_count += 1

                if retry_count > MAX_RETRIES:
                    break
                break

        except openai.APIError as e:
            print(e)
            retry_count += 1
            if retry_count > MAX_RETRIES:
                break

            continue

    query.all_messages = [Message(**x) for x in messages]
    db.add(query)
    db.commit()

    yield query

"""
Slight variant of https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb

NOTICE:
Eyeball testing has determined that this overestimates tokens by ~1% for gpt-4-1106-preview.
"""
def num_tokens_from_messages(messages, model="gpt-4-1106-preview"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in {
        "gpt-3.5-turbo-0613",
        "gpt-3.5-turbo-16k-0613",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-4-1106-preview"
        }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0613")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


