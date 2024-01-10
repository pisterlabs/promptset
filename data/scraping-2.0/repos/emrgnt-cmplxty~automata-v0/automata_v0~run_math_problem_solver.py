import pandas as pd
import json
import os
import numpy as np
from glob import glob
from utils import parse_arguments
from agent.completion_provider import CompletionProvider, RunMode, ProblemType
import openai
import dotenv

dotenv.load_dotenv()

np.random.seed(42)


def load(num_events=10):
    # files = glob(os.path.join("data", "inputs", "MATH", "**", "*.jsonl"))
    # print(f"files = {files}")

    inputs = glob(os.path.join("data", "inputs", "MATH", "*", "*"))
    print(f"len(inputs) = {len(inputs)}")
    indices = list(range(len(inputs)))
    np.random.shuffle(indices)

    results = []
    for index in indices[:num_events]:
        with open(inputs[index], "r") as f:
            results.append(json.loads(f.read()))

    return pd.DataFrame(results)

    # with open("data/inputs/math.txt", "w") as f:
    #     f.write("\n".join([inputs[i] for i in indices[:events]]))
    # return [inputs[i] for i in indices[:events]]


if __name__ == "__main__":
    args = parse_arguments()

    df = load(2)

    print("key = ", os.getenv("OPENAI_API_KEY_LOCAL", ""))
    openai.api_key = os.getenv("OPENAI_API_KEY_LOCAL", "")

    # print(f"df = {df}")
    problem, solution = df.problem[0], df.solution[0]

    completion_provider = CompletionProvider(
        run_mode=RunMode(args.run_mode),
        model=args.model,
        temperature=args.temperature,
        problem_type=ProblemType("math"),
    )

    completion = completion_provider.get_completion(
        task_input=problem, code_snippet=None
    )
    print(f"problem = {problem}")
    print(f"solution = {solution}")
    print(f"completion = {completion}")
