"""Classify the BBC activities to EYFS areas of learning."""

import asyncio
import os
import time

import openai
import pandas as pd

from aiohttp import ClientSession
from dotenv import load_dotenv

from genai import FunctionTemplate
from genai import MessageTemplate
from genai.eyfs import EYFSClassifier
from genai.utils import batch
from genai.utils import create_directory_if_not_exists
from genai.utils import read_json


load_dotenv()

# You need to create this manually before running the script
OUTPUT_FILENAME = "data/"
PATH_TO_AREAS_OF_LEARNING = "src/genai/eyfs/areas_of_learning.json"
PATH_TO_MESSAGE_PROMPT = "src/genai/eyfs/prompts/classifier.json"
PATH_TO_FUNCTION = "src/genai/eyfs/prompts/classifier_function.json"
openai.api_key = os.environ["OPENAI_API_KEY"]


def get_bbc_activities(path: str) -> pd.DataFrame:
    """Read and clean the BBC activities file and return a dataframe."""
    df = pd.read_csv(path)
    df = df.rename(columns={"Age Range (if applicable)": "Age"})
    df = df.dropna(subset=["text", "URL"])
    df = df.drop_duplicates(subset=["URL"])

    return df


def get_areas_of_learning(path: str) -> tuple:
    """Get the EYFS areas of learning and their text."""

    # keys for the classes, text for the prompt
    areas_of_learning = read_json(path)
    areas_of_learning_text = "\n".join([v for v in areas_of_learning.values()])
    areas_of_learning_keys = list(areas_of_learning.keys())

    # Add "None" to cover cases where no option is applicable.
    areas_of_learning_keys.append("None")

    return areas_of_learning_keys, areas_of_learning_text


async def main() -> None:
    """Create prompts for path selection and infer paths."""
    openai.aiosession.set(ClientSession())

    # Fetch the BBC activities
    activities_df = get_bbc_activities(os.environ["PATH_TO_BBC_ACTIVITIES_DATA"])

    # Fetch the EYFS areas of learning
    _, areas_of_learning_text = get_areas_of_learning(PATH_TO_AREAS_OF_LEARNING)

    print(f"Number of BBC activities: {len(activities_df)}")  # noqa: T001

    message = MessageTemplate.load(PATH_TO_MESSAGE_PROMPT)
    function = FunctionTemplate.load(PATH_TO_FUNCTION)
    model = "gpt-3.5-turbo"
    temperature = 0.6

    for i, batched_results in enumerate(batch(activities_df, 20)):
        print(f"Batch {i} / {len(activities_df) // 20}")  # noqa: T001
        tasks = [
            EYFSClassifier.agenerate(
                model=model,
                temperature=temperature,
                messages=[message],
                message_kwargs={
                    "areas_of_learning": areas_of_learning_text,
                    "text": tup.text,
                    "url": tup.URL,
                },
                functions=[function.to_prompt()],
                function_call={"name": "predict_area_of_learning"},
                max_tokens=100,
                concurrency=5,
            )
            for tup in batched_results.itertuples()
        ]

        for future in asyncio.as_completed(tasks):
            result = await future  # Get the result (waits if not ready)
            await EYFSClassifier.write_line_to_file(result, OUTPUT_FILENAME)  # Write to the file

        time.sleep(2)

    await openai.aiosession.get().close()  # Close the http session at the end of the program


if "__main__" == __name__:
    # create data/ directory if it doesn't exist
    create_directory_if_not_exists(OUTPUT_FILENAME)

    start = time.time()
    s = time.perf_counter()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
    finally:
        loop.close()
    e = time.perf_counter()

    print(f"Concurrent execution completed in: {e - s:0.2f} seconds")  # noqa: T001
