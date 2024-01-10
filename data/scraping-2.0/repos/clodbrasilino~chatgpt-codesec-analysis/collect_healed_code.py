import openai
import time

from os import getenv, path
from mbpp import problems
from tqdm import tqdm
from multiprocessing.pool import ThreadPool

openai.api_key = getenv("OPENAI_API_KEY_P")
source_code_path = "./collected_code/to_be_healed/"
collected_code_path = "./collected_code/healed/"


def collect_generated_healed_code() -> None:
    progress = tqdm(total=len(problems))
    with ThreadPool(processes=5) as threadpool:
        for problem in problems:
            threadpool.apply_async(
                call_llm,
                args=(problem, progress),
                callback=llm_result_callback,
            )
        threadpool.close()
        threadpool.join()


def llm_result_callback(result):
    completion = result[0]
    filename = result[1]
    if completion and filename:
        with open(filename, "w+") as f:
            code = str(completion.choices[0].message.content)
            f.write(code)


def call_llm(
    problem: dict[str, str | list[str]],
    progress: tqdm,
    temperature: int = 1,
    top_p: int = 1,
    round: int = 1,
):
    try:
        filename = f"{collected_code_path}/problem-{problem['id']}.txt"
        sourcename = f"{source_code_path}/problem-{problem['id']}.c"
        if path.exists(filename):
            progress.update()
            print(f"{filename} already processed. Skipping.")
            return (None, None)
        if not path.exists(sourcename):
            progress.update()
            print(f"{sourcename} not found. Skipping.")
            return (None, None)
        print(f"{filename} started.")
        with open(sourcename, "r") as weak_source:
            completion = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Acting as an experienced C developer, "
                            "analyze the following source-code: "
                            f"{weak_source.read()}\n "
                            "Re-write the source-code, paying attention to "
                            "the comments to check for fixes for the possible "
                            "weaknesses identified. "
                            "Don't forget to add main function, "
                            "and proper includes and function definitions. "
                            "don't write any comments, just write the code."
                        ),
                    },
                ],
                temperature=temperature,
                top_p=top_p,
            )
        progress.update()
        print(f"{filename} finished.")
        return (completion, filename)
    except KeyboardInterrupt:
        exit(1)
    except BaseException as ex:
        print(ex)
        print(
            f"Problem {problem['id']}, temperature {temperature}, Top P {top_p}, round {round}: Connection failed. Retrying..."
        )
        time.sleep(5)
        return call_llm(problem, temperature, top_p, round)


if __name__ == "__main__":
    collect_generated_healed_code()
