import openai
import time

from os import getenv, path
from mbpp import problems
from tqdm import tqdm
from multiprocessing.pool import ThreadPool

openai.api_key = getenv("OPENAI_API_KEY_P")
collected_code_path = "./collected_code_2"


def collect_generated_code_v2() -> None:
    progress = tqdm(total=len(problems) * 5 * 3 * 5)
    with ThreadPool(processes=20) as threadpool:
        for problem in problems:
            for temperature in range(5):
                for top_p in range(3):
                    for gen_round in range(5):
                        threadpool.apply_async(
                            call_llm,
                            args=(problem, temperature, top_p, gen_round, progress),
                            callback=llm_result_callback,
                        )
        threadpool.close()
        threadpool.join()


def llm_result_callback(result):
    completion = result[0]
    filename = result[1]
    if completion and filename:
        with open(filename, "w") as f:
            code = str(completion.choices[0].message.content)
            f.write(code)


def call_llm(problem, temperature: int, top_p: int, round: int, progress: tqdm):
    try:
        filename = f"{collected_code_path}/problem-{problem['id']}-temp-{temperature/2}-top_p-{top_p/2}-{round+1}.txt"
        if path.exists(filename):
            progress.update()
            return (None, None)
        print(f"{filename[:-2]} started.")
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Acting as an experienced C developer, "
                        f"{problem['text']}"
                        "don't forget to add main function, "
                        "and proper includes and function definitions. "
                        "don't write any comments, just write the code."
                    ),
                },
            ],
            temperature=temperature / 2,
            top_p=top_p / 2,
        )
        progress.update()
        print(f"{filename[:-2]} finished.")
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
    collect_generated_code_v2()
