# server.py
import json
import logging
import os
import pickle
import re
import time

from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel, ConfigDict

app = FastAPI()

SYSTEM_ROLE = os.environ.get("SYSTEM_ROLE")


def clean_text(text):
    # This regex pattern keeps
    # Hangul characters, Latin alphabets, spaces, and the punctuation marks "!" and "?"
    clean_text_pattern = re.compile(
        "[^"
        "\uAC00-\uD7A3"  # Hangul characters
        "a-zA-Z"  # Latin alphabets
        " !?"  # Spaces and the punctuation marks "!" and "?"
        "]+"
    )
    return clean_text_pattern.sub(r"", text)


class ReviewRequest(BaseModel):
    # id: PyObjectId = Field(alias="id")
    content: str

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)


def wait_for_run_completion(client, thread_id, run_id, sleep_interval=5):
    """
    Waits for a run to complete and prints the elapsed time.:param client: The OpenAI client object.
    :param thread_id: The ID of the thread.
    :param run_id: The ID of the run.
    :param sleep_interval: Time in seconds to wait between checks.
    """
    while True:
        try:
            run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
            if run.completed_at:
                elapsed_time = run.completed_at - run.created_at
                formatted_elapsed_time = time.strftime(
                    "%H:%M:%S", time.gmtime(elapsed_time)
                )
                logging.info(f"Run completed in {formatted_elapsed_time}")
                break
        except Exception as e:
            logging.error(f"An error occurred while retrieving the run: {e}")
            break
        logging.info("Waiting for run to complete...")
        time.sleep(sleep_interval)


def assistant():
    message = client.beta.threads.messages.create(
        thread_id=thread.id, role="user", content=review_text
    )
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id="asst_YVHnRUKNjbo1mODGiCEuNIPX",
    )
    wait_for_run_completion(client, thread.id, run.id)
    messages = client.beta.threads.messages.list(thread_id=thread.id)

    last_message = messages.data[0]
    response = last_message.content[0].text.value


def process_review(review):
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    review_text = review.content

    # Filter similarities above threshold K and create a dictionary with the results
    response = client.chat.completions.create(
        response_format={"type": "json_object"},
        model="gpt-3.5-turbo-1106",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_ROLE,
            },
            {
                "role": "user",
                "content": "에센스 토너 필요해서 글로우픽 보고 사봤는데, 좋네여오… 특히 핸드 토너로 사용하고 핸드크림 바르면 완전 흡수력도 짱짱 보습도 짱짱..",
            },
            {
                "role": "assistant",
                "content": '{"nlpAnalysis":{"O":0.4,"D":0.4,"R":0.4,"S":0.4,"P":0.4,"N":0.4,"T":0.4,"W":0.4}}',  # less tokens I hope
            },
            {
                "role": "user",
                "content": "너무 안좋은데요?",
            },
            {
                "role": "assistant",
                "content": '{"nlpAnalysis":{"O":0.0,"D":0.0,"R":0.0,"S":0.0,"P":0.0,"N":0.0,"T":0.0,"W":0.0}}',  # less tokens I hope
            },
            {
                "role": "user",
                "content": clean_text(review_text),  # less tokens I hope
            },
        ],
        temperature=1,
        max_tokens=128,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    nlpanalysis = json.loads(response.choices[0].message.content)

    return {
        "nlpAnalysis": nlpanalysis["nlpAnalysis"],
    }


@app.post("/process-review")
async def process_review_endpoint(review_request: ReviewRequest):
    nlp_result = process_review(review_request)
    return nlp_result


@app.get("/health")
def health_check():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("__main__:app", host="0.0.0.0", port=9000)
