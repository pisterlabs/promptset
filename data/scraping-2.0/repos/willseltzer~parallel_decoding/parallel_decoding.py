import asyncio
import os
import time
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import aiohttp
import openai
import pandas as pd
import tiktoken


class CallType(Enum):
    PARALLEL = "parallel"
    NORMAL = "normal"

    def __lt__(self, other):
        if isinstance(other, CallType):
            return self.value < other.value
        return NotImplemented


@dataclass
class Result:
    prompt: str
    model_response: str
    model: str
    time_seconds: float
    openai_call_type: CallType
    tokens_per_second: Optional[float] = None

    def __post_init__(self):
        self.tokens_per_second = self.calculate_tokens_per_second()

    def calculate_tokens_per_second(self) -> float:
        """Calculate the number of tokens per second."""
        encoding = tiktoken.encoding_for_model(self.model)
        encoded = encoding.encode(self.model_response)
        return len(encoded) / self.time_seconds

def print_enclosed_in_lines(text: str) -> None:
    """Enclose the given text in lines."""
    print("\n" + "\n".join(["-" * (len(text)),  text, "-" * (len(text))]))



def get_skeleton_template(prompt: str) -> str:
    """Generate skeleton template for the given question."""

    skeleton_template = """
    You’re an organizer responsible for only giving the skeleton (not the full content) for answering the question.
    Provide the skeleton in a list of points (numbered 1., 2., 3., etc.) to answer the question. Instead of writing a full
    sentence, each skeleton point should be very short with only 3∼5 words.
    Question:
    What are the typical types of Chinese dishes?
    Skeleton:
    1. Dumplings.
    2. Noodles.
    3. Dim Sum.
    4. Hot Pot.
    5. Wonton.
    6. Ma Po Tofu.
    7. Char Siu.
    8. Fried Rice.
    Question:
    What are some practical tips for individuals to reduce their carbon emissions?
    Skeleton:
    1. Energy conservation.
    2. Efficient transportation.
    3. Home energy efficiency.
    4. Reduce water consumption.
    5. Sustainable diet.
    6. Sustainable travel.
    Now, please provide the skeleton for the following question.
    {prompt}
    """
    return skeleton_template.format(prompt=prompt)


def get_point_expanding_template(prompt: str, skeleton: str, subtask: str) -> str:
    """Generate point expanding template for the given question, skeleton, and subtask."""
    point_expanding_template = """
    You’re responsible for continuing the writing of one and only one point in the overall answer to the following
    question.
    {prompt}
    The outline of the full answer is:
    {skeleton}

    Continue the above and only continue the writing of point {subtask}. Expand on it in a single sentence and do not continue with other points!
    """

    return point_expanding_template.format(
        prompt=prompt, skeleton=skeleton, subtask=subtask
    )



async def openai_async(messages: List[str], model) -> str:
    resp = await openai.ChatCompletion.acreate(messages=messages, model=model)
    return resp["choices"][0]["message"]["content"]

def openai_sync(messages: List[str], model) -> str:
    resp =  openai.ChatCompletion.create(messages=messages, model=model)
    return resp["choices"][0]["message"]["content"]

async def parallel_coroutine(
    subtask: List[str],
    prompt: str,
    skeleton: str,
    model: str
) -> List[str]:
    
    prompts: List[List[dict]]=[[{"content":get_point_expanding_template(prompt, skeleton, subtask), "role": "user"}] for subtask in subtask]
    coroutines = [
        openai_async(messages=prompt, model=model)
        for prompt in prompts
    ]
    return await asyncio.gather(*coroutines)


async def main(prompt: str, num_iterations: int, model: str) -> List[Result]:
    async with aiohttp.ClientSession() as session:
        openai.aiosession.set(session)
        results = []
        for _ in range(num_iterations):
            parallel_res = await run_parallel(prompt=prompt, model=model)
            results.append(parallel_res)
            results.append(run_normal(prompt=prompt, model=model))
        return results


def run_normal(prompt: str, model: str) -> Result:
    print("\nCOMPUTING NORMAL RESPONSE...")
    start = time.time()
    try:
        model_response: str = openai_sync(messages=[{"content": prompt, "role": "user"}], model=model)
        print_enclosed_in_lines("NORMAL RESPONSE")
        print(model_response)
    except Exception as e:
        print(f"An error occurred: {e}")
        return
    end = time.time()
    return Result(
        prompt=prompt,
        model_response=model_response,
        model=model,
        time_seconds=end - start,
        openai_call_type=CallType.NORMAL,
    )


async def run_parallel(prompt: str, model: str) -> Result:
    skeleton = get_skeleton_template(prompt=prompt)
    start = time.time()
    try:
        skeleton_response: str = openai_sync(messages=[{"content": skeleton, "role": "user"}], model=model)
        print_enclosed_in_lines("SKELETON")
        print(skeleton_response)
        total_resp = [x for x in skeleton_response]
        subtask: List[str] = "".join(total_resp).split("\n")
        responses: List[str] = await parallel_coroutine(
                subtask=subtask,
                prompt=prompt,
                skeleton=skeleton,
                model=model
            )
        # sort the responses by number
        responses = map(lambda s: s.strip("Skeleton:").lstrip(), responses)
        sorted_responses = sorted(responses, key=lambda s: int(s.split(".")[0]))
        parallel_response = "\n".join(sorted_responses)
        print_enclosed_in_lines("PARALLEL RESPONSE")
        print(parallel_response)
    except Exception as e:
        print(f"An error occurred: {e}")
        return
    end = time.time()

    return Result(
        prompt=prompt,
        model_response=parallel_response,
        model=model,
        time_seconds=end - start,
        openai_call_type=CallType.PARALLEL,
    )


if __name__ == "__main__":
    openai.api_key = os.getenv("OPENAI_API_KEY")
    prompt = "Give me five ideas of mediterranean food I can cook for dinner."
    print_enclosed_in_lines("QUERY")
    print(prompt)
    results = asyncio.run(main(prompt=prompt, num_iterations=1, model="gpt-4"))
    df = pd.DataFrame(results)
    desc = df[['tokens_per_second', 'openai_call_type']].groupby("openai_call_type").describe()
    print_enclosed_in_lines("TOKENS PER SECOND")
    print(desc['tokens_per_second'][['min', '25%','50%','75%', 'max']])
