"""Define gpt3_api: wrap openai.Completion.create.

set OPENAI_API_KEY=your_openai_key
curl -H "Authorization: Bearer %OPENAI_API_KEY%" https://api.openai.com/v1/engines
# linux and friends:
# export OPENAI_API_KEY=your_openai_key
# curl -H "Authorization: Bearer $OPENAI_API_KEY" https://api.openai.com/v1/engines

https://beta.openai.com/docs/engines/instruct-series-beta
    davinci-instruct-beta, curie-instruct-beta, and our newest addition, davinci-instruct-beta-v3
"""
# pylint: disable=too-many-arguments, too-many-locals

from collections import deque
from typing import List, Tuple  # Deque,

# from itertools import chain
import logzero
import openai
from logzero import logger
from set_loglevel import set_loglevel

from gpt3_api.config import Settings

config = Settings()
openai.api_key = config.api_key

# logzero.loglevel(10)
logzero.loglevel(set_loglevel())


def assemble_prompt(
    query: str = "",
    preamble: str = "",
    prefixes: Tuple[str, str] = ("Human: ", "AI: "),
    suffixes: Tuple[str, str] = ("\n", "\n\n"),
    # examples: List[Tuple[str, str]] = [("", "")],
    examples: List[Tuple[str, str]] = None,
) -> str:
    """Assemble prompt."""
    if examples:
        _ = "".join(
            [
                prefixes[0] + elm[0] + suffixes[0] + prefixes[1] + elm[1] + suffixes[1]
                for elm in examples
            ]
        )
    else:
        _ = ""
    _ = preamble + suffixes[1] + _ + prefixes[0] + query + suffixes[0] + prefixes[1]

    # remove spaces in both ends
    return _.strip()


# fmt: off
def get_resp(
        prompt: str,
        engine: str = "davinci",
        temperature: float = 0.9,
        max_tokens: int = 150,
        top_p: int = 1,
        frequency_penalty: float = 0.0,  # -2..2, postive penalty
        presence_penalty: float = 0.0,
        stop=None,
        # **kwargs,
) -> str:
    # fmt: on
    """Get response from openai.

    query = "this test"
    preamble = "Translatiton"
    prefixes = ("English: ", "中文: ")
    suffixes = ("\n", "\n\n")
    examples = [
        ("Good solution by Vengat, and this also works with rjust.", "Vengat 提供了很好的解决方案，这也适用于 rjust。"),
        # ("Good morning", "早上好"),
        # ("I love you", "我爱你"),
        ("I have tried to use all GPT-3 engines to reach the results and the only that gives back an accurate result is DAVINCI.", "我尝试使用所有 GPT-3 引擎来获得结果，唯一能返回准确结果的是 DAVINCI。")
    ]

    _ = "".join([prefixes[0] + elm[0] + suffixes[0] + prefixes[1] + elm[1] + suffixes[1] for elm in examples])
    prompt = preamble +suffixes[1] + _ + prefixes[0] + query + suffixes[0] + prefixes[1]

    engine: str = "davinci"
    temperature: float = 0.2
    max_tokens: int = 150
    top_p: int = 1
    frequency_penalty: float = 0.0  # -2..2, postive penalty
    presence_penalty: float = 0.0
    stop = None
    stop = suffixes[1]

    """
    # turn of echo for openai.Completion.create when debug is on
    echo = False
    if logger.level < 20:
        echo = True

    logger.debug("prompt: [[%s]]", prompt)

    try:
        # https://beta.openai.com/docs/api-reference/completions/create
        response = openai.Completion.create(
            prompt=prompt,
            engine=engine,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,  # -2.0 and 2.0. Positive values penalize new tokens
            presence_penalty=presence_penalty,
            echo=echo,
            stop=stop,   # default to null
            # **kwargs,
        )
        logger.debug(response)
    except Exception as exc:
        logger.error(exc)
        return str(exc)

    try:
        resp = response.choices[0].text
        get_resp.response = response
    except Exception as exc:
        logger.error(exc)
        resp = str(exc)

    return resp


# fmt: off
def gpt3_api(
        query: str,
        prompt: str = "",
        engine: str = "davinci",  # davinci, curie, babbage, ada
        # prompt="The following is a conversation with an AI assistant. The assistant is helpful, creative, clever, and very friendly.\n\nHuman: Hello, who are you?\nAI: I am an AI created by OpenAI. How can I help you today?\nHuman: I'd like to cancel my subscription.\nAI:",  # pylint: disable=line-too-long
        temperature: float = 0.9,
        max_tokens: int = 150,
        top_p: int = 1,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.6,
        stop=None,
        prefixes: Tuple[str, str] = ("Human: ", "AI: "),
        suffixes: Tuple[str, str] = ("\n", "\n\n"),
        chat_mode: bool = False,
        preamble: str = "",
        # examples: List[Tuple[str, str]] = [("", "")],
        examples: List[Tuple[str, str]] = None,
        deq_len: int = 100,
        # proxy: str = None,
        # proxy: Optional[str] = None,
        # **kwargs,
) -> str:
    # fmt: on
    """Define api.

    Args:
        prefixes: prefix0, prefix1
        suffixes: suffix0, suffix1
            to form the prompt
            preamble + suffix1 + prefix0 + query + suffix0 + prefix1 + result + suffix1 + prefix0 + query + suffix0 + prefix1
        preamble: use when priming
        examples: list of (query, result) pairs
        deq_len: queue length

        openai.completion args: check [openai docs](https://beta.openai.com/docs/api-reference/completions/create)
        engine: https://beta.openai.com/docs/engines
        temperature:

    Returns:
        response = openai.Completion()
        response.choices[0].text
    """
    prefix0, prefix1 = prefixes
    suffix0, suffix1 = suffixes

    # may use your own prompt any time
    if prompt:
        # make sure stop == suffixes[1]
        if not stop == suffixes[1]:
            logger.warning("stop != suffixes[1]: you may have a problem unless you know what you are doing.")

        return get_resp(
            prompt=prompt,
            engine=engine,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=stop,
            # **kwargs,
        )
    try:
        query = str(query)
    except Exception:
        query = ""
    if not query.strip():
        logger.warning(" query (%s) is empty. Make sure this is really what you want.", query)

    # initialize deq and store logged data as function attribute.
    # can be useful in chatbot
    try:
        _ = gpt3_api.deq
    except AttributeError:
        # only run on first call
        if examples:
            gpt3_api.deq = deque(examples, deq_len)
        else:
            gpt3_api.deq = deque([], deq_len)

    if chat_mode:  # use past log (deq), eg chat
        # _ = f"{suffix0}{prefix1}".join(gpt3_api.deq)
        prompt = assemble_prompt(
            query,
            preamble=preamble,
            examples=list(gpt3_api.deq),
        )
    else:  # normal op, always use preamble and examples
        # construct prompt when not provided by user,
        # not bool(prompt) == True, non empty prompt already handled:
        prompt = assemble_prompt(
            query,
            preamble=preamble,
            examples=examples,
        )

    resp = get_resp(
        prompt=prompt,
        engine=engine,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop,
        # **kwargs,
    )

    # update deq if resp is not empty
    if not resp.strip():
        gpt3_api.deq.append((query, resp))

    return resp


def main():
    """Run some test codes."""
    # preamble = "Tell a joke."
    # preamble = ""

    preamble = "Translatiton"
    prefixes = ("English: ", "中文: ")
    suffixes = ("\n", "\n\n")

    temperature = 0.5
    frequency_penalty = 0
    presence_penalty = 0

    examples = [
        ("Good solution by Vengat, and this also works with rjust.", "Vengat 提供了很好的解决方案，这也适用于 rjust。"),
        # ("Good morning", "早上好"),
        # ("I love you", "我爱你"),
        ("I have tried to use all GPT-3 engines to reach the results and the only that gives back an accurate result is DAVINCI.", "我尝试使用所有 GPT-3 引擎来获得结果，唯一能返回准确结果的是 DAVINCI。")
    ]
    del examples

    # "Hi there!",
    "I hate you!",
    # "Good morning",
    # "Good solution by Vengat, and this also works with rjust.",
    query = "What's your name?",
    # query = "I have tried to use all GPT-3 engines to reach the results and the only that gives back an accurate result is DAVINCI.",

    while True:
        query = "Make sure you have a valid API Token to use GPT-3."
        query = input("English: ")
        if query.strip().lower()[:4] in ["quit", "exit", "stop"]:
            break

        _ = gpt3_api(
            query,
            preamble=preamble,
            prefixes=prefixes,
            suffixes=suffixes,
            temperature=temperature,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
        )
        print("中文： ", _)


if __name__ == "__main__":
    main()
