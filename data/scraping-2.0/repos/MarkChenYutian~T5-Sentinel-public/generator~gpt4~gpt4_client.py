"""
@brief: A GPT4 response generator using Async io
@author: Yutian Chen <yutianch@andrew.cmu.edu>
@date: May 15, 2023
"""

import asyncio
import openai
import json
import yaml
import time

from pathlib import Path
from generator.chatgpt.chatgpt_client import \
    chatgpt_pred_fn, chatgpt_task_generator, chatgpt_state_initializer, estimate_token_count, HANDLE_STRATEGY
from generator.client_base import AsyncRequestClient, TaskResult
import pipeline.component.text_component as P

Converter = P.WriteExtra({"source": "gpt4"}) >> P.ToJsonStr()

async def gpt4_request_fn(self: AsyncRequestClient, state, subset, uid, text) -> TaskResult:
    if state["token"] > self.config["MaxTokenCount"]:
        print("Abort due to budget limit.")
        raise Exception("Exceed the MaxTokenCount setting")

    await self.worker_lock.acquire()
    start_time = time.time()

    # Ready ... now Work!

    estimatedNumTokens = estimate_token_count(text)
    if estimatedNumTokens > self.config["MaxLengthAllowed"]:
        print("[x]\t", uid,
              "failed since it exceeds the token limit (" + str(self.config["MaxLengthAllowed"]) + ")")
        self.worker_lock.release()
        return TaskResult.CANCEL

    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[
                {"role": "user", "content": "Rephrase the following paragraph by paragraph:\n\n" + text}
            ]
        )

    except openai.error.InvalidRequestError:
        # no need to wait, since the request is not sent for some reason
        await asyncio.sleep(1.0)  # Avoid flushing the API
        self.worker_lock.release()
        return TaskResult.RETRY

    except (openai.error.RateLimitError, openai.error.APIError, openai.error.TryAgain, openai.error.Timeout):
        await asyncio.sleep(self.config["WaitTime"])
        self.worker_lock.release()
        return TaskResult.RETRY

    finishReason = response["choices"][0]["finish_reason"]
    result = HANDLE_STRATEGY[finishReason]

    if result == TaskResult.FINISH:
        machineText = response["choices"][0]["message"]["content"].strip()

        await self.writer_lock.acquire()
        with open(Path(self.config["OutputDirectory"], subset + ".jsonl"), "a", encoding="utf-8") as f:
            f.write(Converter({"uid": uid, "text": machineText, "extra": dict()}))
            f.write("\n")
        self.writer_lock.release()
        self.state["processed"].add((subset, uid))

    self.state["token"] += response["usage"]["total_tokens"]

    # Wait for 60 secs, then release the lock to spawn a new worker coroutine
    # (We won't be blocked out)
    end_time = time.time()
    await asyncio.sleep(self.config["WaitTime"] - (end_time - start_time))
    self.worker_lock.release()

    return result

if __name__ == "__main__":
    with open("./generator/gpt4/gpt4_client.yaml", "r") as f:
        chatgpt_config = yaml.safe_load(f)

    with open(Path(chatgpt_config["ClientRoot"], "secret.json"), "r") as f:
        API_KEY = json.load(f)["OPENAI_API_KEY"]
        openai.api_key = API_KEY

    ChatGPTClient = AsyncRequestClient(
        chatgpt_config,
        gpt4_request_fn,
        chatgpt_pred_fn,
        chatgpt_task_generator,
        chatgpt_state_initializer,
        display_args=lambda args: args[1]
    )
    asyncio.run(ChatGPTClient.execute())
