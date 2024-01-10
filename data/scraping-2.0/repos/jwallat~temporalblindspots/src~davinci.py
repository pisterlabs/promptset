import os
import logging
import openai
import pandas as pd
import time
from tqdm import tqdm
import multiprocessing as mp
import datetime
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


@retry(wait=wait_random_exponential(min=3, max=60), stop=stop_after_attempt(8))
def make_completion_request(model_name, prompt, question):
    # start_sequence = "\nA:"
    # restart_sequence = "\n\nQ: "

    input = prompt.replace("{question}", question)
    # logging.info(f"input: {input}")

    response = openai.Completion.create(
        model=model_name,
        prompt=input,
        temperature=0,
        max_tokens=50,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=["\n"],
    )

    return response


def get_davinci_completions(model_name, data, run_name, prompt, args):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    job_df = {"q_id": [], "question": [], "answer": []}

    # print(data.head())
    for index in tqdm(range(0, len(data))):
        question = data["Question"][index]

        try:
            # if index % 50 == 0 and index != 0:
            #     print("************", index)
            #     time.sleep(10)
            response = make_completion_request(model_name, prompt, question)

            if response.choices[0].text.strip() == "":
                job_df["answer"].append("<no response>")
            else:
                job_df["answer"].append(response.choices[0].text.strip())
        except Exception as e:
            print("Could not get response. Here is the exception:", str(e))
            job_df["answer"].append("<exception>")

        job_df["q_id"].append(index)
        job_df["question"].append(question)
        # logging.info("Got response: ", response)
        # logging.info("\n\n\n")

        final_df = pd.DataFrame(job_df)
        final_df.to_csv(f"{run_name}_predictions.csv", index=False, sep="\t")
    return final_df


def get_davinci_completions_mp(model_name, data, run_name, prompt):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    # job_df = {"q_id": [], "question": [], "answer": []}

    now = datetime.datetime.now()
    timestamp = now.strftime("%Y_%m_%dT%H:%M")
    file_name = f"{run_name}_predictions_{timestamp}.csv"

    manager = mp.Manager()
    q = manager.Queue()
    file_pool = mp.Pool(1)
    file_pool.apply_async(listener, (q, file_name))

    pool = mp.Pool(15)
    jobs = []

    for index in tqdm(range(0, len(data))):
        question = data["Question"][index]

        job = pool.apply_async(handle_qa_pair, (index, question, model_name, prompt, q))
        jobs.append(job)

    for job in tqdm(jobs):
        job.get()

    q.put("#done#")  # all workers are done, we close the output file
    pool.close()
    pool.join()

    # order the results
    df = pd.read_csv(file_name, sep="\t")
    final_df = df.sort_values(by=["q_id"])
    final_df.to_csv(file_name + "_ordered", index=False, sep="\t")

    return final_df


def handle_qa_pair(index, question, model_name, prompt, q):
    print("Handling qa pair: ", index)
    try:
        if index % 3 == 0 and index != 0:
            print("************", index)
            time.sleep(10)
        response = make_completion_request(model_name, prompt, question)

        if response.choices[0].text.strip() == "":
            answer = "<no response>"
        else:
            answer = response.choices[0].text.strip()
        # answer = response
    except Exception as e:
        print("Could not get response. Here is the exception:", str(e))
        answer = "<exception>"

    # Put response into queue
    q.put(f"{index}\t{question}\t{answer}\n")


def listener(q, file_name):
    """
    continue to listen for messages on the queue and writes to file when receive one
    if it receives a '#done#' message it will exit
    """
    with open(file_name, "a") as f:
        f.write("q_id\tquestion\tanswer\n")
        f.flush()
        while True:
            m = q.get()

            if m == "#done#":
                print("Got break")
                break
            f.write(m)
            f.flush()
