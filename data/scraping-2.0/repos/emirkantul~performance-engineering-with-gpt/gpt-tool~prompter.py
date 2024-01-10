import openai
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set the API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key is None:
    raise ValueError("OPENAI_API_KEY environment variable not set")

openai.api_key = openai_api_key


MODEL_NAME = "gpt-3.5-turbo"

import time


def handle_rate_limit(func):
    def wrapper(*args, **kwargs):
        attempts = 0
        while attempts < 3:
            try:
                return func(*args, **kwargs)
            except openai.error.RateLimitError:
                wait_time = (attempts + 1) * 5
                print(f"RateLimit error. Waiting for {wait_time} seconds.")
                time.sleep(wait_time)
                attempts += 1
        print("Exceeded rate limit retry attempts. Exiting.")
        return None

    return wrapper


@handle_rate_limit
def get_fix(last_resp):
    previous_code = last_resp["code"]
    previous_perf = last_resp["perf_results"]

    prompt = (
        "Take the persona of a HPC expert who can fix  parallel and"
        " sequential code for better performance. Given old code, gpu"
        " architecture, cpu architecture, found performance issues and"
        " performance results fix this code and write what has been"
        f" changed:\nPrevious Code: {previous_code}\nPrevious Perf Results:"
        f" {previous_perf}\nPerformance Issues:"
        f" {last_resp['performance_issues']}\nGPU Architecture:"
        f" {last_resp['gpu']}\nCPU Architecture: {last_resp['cpu']}\nWRITE"
        " YOUR ANSWERS IN THIS FORMAT:\nNew Code: {new_code}\nWhat"
        " Changed: {what_changed}\n\n"
    )
    print(f"PROMPT: {prompt}")

    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        temperature=0.5,
        n=1,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a HPC expert who can analyze parallel and"
                    " sequential code and fix performance bugs."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    result = response["choices"][0]["message"]["content"]
    print(f"RESPONSE: {result}")

    new_code, whats_changed = result.split("What Changed:")
    new_code = new_code.strip()
    new_code = new_code.replace("New Code:", "")
    whats_changed = whats_changed.strip()

    return {"new_code": new_code, "whats_changed": whats_changed}


@handle_rate_limit
def get_architecture_recommendation(code, cpu, gpu):
    prompt = (
        "Take the persona of a HPC expert who can give best recommendations"
        " for given cpu and gpu architecture. Give the best practices user"
        " can achieve with this architecture and code. Only give"
        f" recommendation NO CODE\n\nCPU: {cpu}\nGPU: {gpu}\n\nCode:\n{code}"
    )
    print(f"PROMPT: {prompt}")

    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        temperature=0.5,
        n=1,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a HPC expert who can analyze parallel and"
                    " sequential code and classify performance bugs."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    result = response["choices"][0]["message"]["content"]
    print(f"RESPONSE: {result}")

    return result


@handle_rate_limit
def get_performance_issues(code):
    prompt = (
        "Given performance issues:\n\nComputational Expensive"
        " Operation\nInefficient Data Structures\nNot Using Function"
        " Inlining\nInefficient Concurrency Control\nMissing SIMD"
        " Parallelism\nMissing GPU Parallelism\nMissing Task"
        " Parallelism\n\nDetect and classify performance-related bugs in"
        " given C++ code - which can be sequential, OpenMP-based (CPU"
        " parallel) or CUDA-based (GPU Parallel).Only use the classes that"
        " are given and ONLY USE THE FORMAT BELOW\nComputational Expensive"
        f" Operation: {{ YOUR ANSWER }}\n...\n\ngiven code:\n{code}"
    )
    print(f"PROMPT: {prompt}")

    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        temperature=0.1,
        n=1,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a HPC expert who can analyze parallel and"
                    " sequential code and classify performance bugs."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    result = response["choices"][0]["message"]["content"]
    print(f"RESPONSE: {result}")

    # Split the response into separate parts for each issue
    issues = result.split("\n")
    issues_dict = {}
    for issue in issues:
        if ": " in issue:
            key, val = issue.split(": ", 1)
            issues_dict[key] = val
    return issues_dict

    return result


@handle_rate_limit
def get_perf_assessment(code, output):
    prompt = (
        "Take the persona of a HPC expert who can give best recommendations"
        " for given 'perf' and compile results. Give the best practices user"
        " can achieve with this code, 'perf', compile and run"
        f" output:.\n\nCode: {code}\nCompile: {output['compile']}\nRun:"
        f" {output['run']}\nPerf: {output['perf']}"
    )
    print(f"PROMPT: {prompt}")

    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        temperature=0.5,
        n=1,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a HPC expert who can analyze parallel and"
                    " sequential code and classify performance bugs."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    result = response["choices"][0]["message"]["content"]
    print(f"RESPONSE: {result}")

    return result


@handle_rate_limit
def get_nvprof_assessment(code, output):
    prompt = (
        "Take the persona of a HPC expert who can give best recommendations"
        " for given 'nvprof' and compile results. Give the best practices"
        " user can achieve with this code,'nvprof', compile and run"
        f" output:.\n\nCode: {code}\nCompile: {output['compile']}\nRun:"
        f" {output['run']}\nNvprof: {output['nvprof']}"
    )
    print(f"PROMPT: {prompt}")

    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        temperature=0.5,
        n=1,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a HPC expert who can analyze parallel and"
                    " sequential code and classify performance bugs."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    result = response["choices"][0]["message"]["content"]
    print(f"RESPONSE: {result}")

    return result
