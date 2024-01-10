from collections import defaultdict
import os
import random
import time
import json

import openai
import pandas as pd
from tqdm import tqdm

from utils import load_dataset


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 3,
    errors: tuple = (openai.RateLimitError,),
):
    """Retry a function with exponential backoff."""

    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay

        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)

            # Retry on specific errors
            except errors:
                # Increment retries
                num_retries += 1

                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )

                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())
                print(f"Retrying in {delay:.2f} seconds...")

                # Sleep for the delay
                time.sleep(delay)

            # Raise exceptions for any errors not specified
            except Exception as e:
                print(f"Error occurred: {e}")
                raise e

    return wrapper


@retry_with_exponential_backoff
def request_completion(
    client: openai.OpenAI,
    model: str,
    prompt_messages: list,
):
    return client.chat.completions.create(
        model=model,
        messages=prompt_messages,
    )


def perform_inference(dataset, test_file_path, model_type="gpt-3.5-turbo-1106"):
    client = openai.OpenAI()

    # 추론 결과 저장
    result_dir = os.path.join(
        os.path.join("results", model_type),
        test_file_path.split("2024_")[-1].split(".json")[0],
    )
    os.makedirs(result_dir, exist_ok=True)

    outputs = defaultdict(list)

    with open(os.path.join(result_dir, "result.jsonl"), "w", encoding="utf-8") as f:
        for i, example in enumerate(tqdm(dataset, total=len(dataset))):
            answer = example["answer"]
            score = example["score"]
            try:
                response = request_completion(
                    client=client,
                    model=model_type,
                    prompt_messages=example["messages"],
                )
                output = response.choices[0].message.content.strip()

                f.write(
                    json.dumps(
                        {
                            "id": i + 1,
                            "output": output,
                            "answer": answer,
                            "score": score,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                outputs["id"].append(i + 1)
                outputs["output"].append(output)
                outputs["answer"].append(answer)
                outputs["score"].append(score)
            except Exception as e:
                print(f"Error occurred: {e}")
                outputs["id"].append(i + 1)
                outputs["output"].append("")
                outputs["answer"].append(answer)
                outputs["score"].append(score)

    pd.DataFrame(outputs).to_csv(
        os.path.join(result_dir, "result.csv"), encoding="utf-8"
    )


# def process_model(model_type: str, test_file_path: str, prompt_type: str):
def process_model(model_type: str, test_file_path: str):
    # 데이터셋 로드
    # dataset = load_dataset(test_file_path=test_file_path, prompt_type=prompt_type)
    dataset = load_dataset(model_type=model_type, test_file_path=test_file_path)

    # 추론 수행
    perform_inference(
        dataset=dataset, test_file_path=test_file_path, model_type=model_type
    )
