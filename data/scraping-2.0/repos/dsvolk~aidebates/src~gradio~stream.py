from collections.abc import Generator
from queue import Empty, Queue
from threading import Thread
from typing import Dict

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from .callbacks import QueueCallback


def stream(prompt_template: ChatPromptTemplate, llm_params: Dict, invoke_params: Dict) -> Generator:
    q: Queue = Queue()
    job_done = object()

    chain = prompt_template | ChatOpenAI(callbacks=[QueueCallback(q)], **llm_params)

    # Create a funciton to call - this will run in a thread
    def task():
        _ = chain.invoke(invoke_params)
        q.put(job_done)

    # Create a thread and start the function
    t = Thread(target=task)
    t.start()

    content = ""

    # Get each new token from the queue and yield for our generator
    while True:
        try:
            next_token = q.get(True, timeout=1)
            if next_token is job_done:
                break
            content += next_token
            yield content
        except Empty:
            continue
