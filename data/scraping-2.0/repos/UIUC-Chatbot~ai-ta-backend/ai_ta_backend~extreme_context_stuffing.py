"""
API REQUEST PARALLEL PROCESSOR

Using the OpenAI API to process lots of text quickly takes some care.
If you trickle in a million API requests one by one, they'll take days to complete.
If you flood a million API requests in parallel, they'll exceed the rate limits and fail with errors.
To maximize throughput, parallel requests need to be throttled to stay under rate limits.

This script parallelizes requests to the OpenAI API while throttling to stay under rate limits.

Features:
- Streams requests from file, to avoid running out of memory for giant jobs
- Makes requests concurrently, to maximize throughput
- Throttles request and token usage, to stay under rate limits
- Retries failed requests up to {max_attempts} times, to avoid missing data
- Logs errors, to diagnose problems with requests

Example command to call script:
```
python examples/api_request_parallel_processor.py \
  --requests_filepath examples/data/example_requests_to_parallel_process.jsonl \
  --save_filepath examples/data/example_requests_to_parallel_process_results.jsonl \
  --request_url https://api.openai.com/v1/embeddings \
  --max_requests_per_minute 1500 \
  --max_tokens_per_minute 6250000 \
  --token_encoding_name cl100k_base \
  --max_attempts 5 \
  --logging_level 20
```

Inputs:
- requests_filepath : str
    - path to the file containing the requests to be processed
    - file should be a jsonl file, where each line is a json object with API parameters and an optional metadata field
    - e.g., {"model": "text-embedding-ada-002", "input": "embed me", "metadata": {"row_id": 1}}
    - as with all jsonl files, take care that newlines in the content are properly escaped (json.dumps does this automatically)
    - an example file is provided at examples/data/example_requests_to_parallel_process.jsonl
    - the code to generate the example file is appended to the bottom of this script
- save_filepath : str, optional
    - path to the file where the results will be saved
    - file will be a jsonl file, where each line is an array with the original request plus the API response
    - e.g., [{"model": "text-embedding-ada-002", "input": "embed me"}, {...}]
    - if omitted, results will be saved to {requests_filename}_results.jsonl
- request_url : str, optional
    - URL of the API endpoint to call
    - if omitted, will default to "https://api.openai.com/v1/embeddings"
- api_key : str, optional
    - API key to use
    - if omitted, the script will attempt to read it from an environment variable {os.getenv("OPENAI_API_KEY")}
- max_requests_per_minute : float, optional
    - target number of requests to make per minute (will make less if limited by tokens)
    - leave headroom by setting this to 50% or 75% of your limit
    - if requests are limiting you, try batching multiple embeddings or completions into one request
    - if omitted, will default to 1,500
- max_tokens_per_minute : float, optional
    - target number of tokens to use per minute (will use less if limited by requests)
    - leave headroom by setting this to 50% or 75% of your limit
    - if omitted, will default to 125,000
- token_encoding_name : str, optional
    - name of the token encoding used, as defined in the `tiktoken` package
    - if omitted, will default to "cl100k_base" (used by `text-embedding-ada-002`)
- max_attempts : int, optional
    - number of times to retry a failed request before giving up
    - if omitted, will default to 5
- logging_level : int, optional
    - level of logging to use; higher numbers will log fewer messages
    - 40 = ERROR; will log only when requests fail after all retries
    - 30 = WARNING; will log when requests his rate limits or other errors
    - 20 = INFO; will log when requests start and the status at finish
    - 10 = DEBUG; will log various things as the loop runs to see when they occur
    - if omitted, will default to 20 (INFO).

The script is structured as follows:
    - Imports
    - Define main()
        - Initialize things
        - In main loop:
            - Get next request if one is not already waiting for capacity
            - Update available token & request capacity
            - If enough capacity available, call API
            - The loop pauses if a rate limit error is hit
            - The loop breaks when no tasks remain
    - Define dataclasses
        - StatusTracker (stores script metadata counters; only one instance is created)
        - APIRequest (stores API inputs, outputs, metadata; one method to call API)
    - Define functions
        - api_endpoint_from_url (extracts API endpoint from request URL)
        - append_to_jsonl (writes to results file)
        - num_tokens_consumed_from_request (bigger function to infer token usage from request)
        - task_id_generator_function (yields 1, 2, 3, ...)
    - Run main()
"""

# import argparse
# import subprocess
# import tempfile
# from langchain.llms import OpenAI
import asyncio
import json
import logging

# import os
import re
import time

# for storing API inputs, outputs, and metadata
from dataclasses import dataclass, field
from typing import Any, List

import aiohttp  # for making API calls concurrently
import tiktoken  # for counting tokens

# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import Qdrant
# from qdrant_client import QdrantClient, models


class OpenAIAPIProcessor:

  def __init__(self, input_prompts_list, request_url, api_key, max_requests_per_minute, max_tokens_per_minute,
               token_encoding_name, max_attempts, logging_level):
    self.request_url = request_url
    self.api_key = api_key
    self.max_requests_per_minute = max_requests_per_minute
    self.max_tokens_per_minute = max_tokens_per_minute
    self.token_encoding_name = token_encoding_name
    self.max_attempts = max_attempts
    self.logging_level = logging_level
    self.input_prompts_list: List[dict] = input_prompts_list
    self.results = []
    self.cleaned_results: List[str] = []

  async def process_api_requests_from_file(self):
    """Processes API requests in parallel, throttling to stay under rate limits."""
    # constants
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = 0.001  # 1 ms limits max throughput to 1,000 requests per second

    # initialize logging
    logging.basicConfig(level=self.logging_level)
    logging.debug(f"Logging initialized at level {self.logging_level}")

    # infer API endpoint and construct request header
    api_endpoint = api_endpoint_from_url(self.request_url)
    request_header = {"Authorization": f"Bearer {self.api_key}"}

    # initialize trackers
    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator = task_id_generator_function()  # generates integer IDs of 1, 2, 3, ...
    status_tracker = StatusTracker()  # single instance to track a collection of variables
    next_request = None  # variable to hold the next request to call

    # initialize available capacity counts
    available_request_capacity = self.max_requests_per_minute
    available_token_capacity = self.max_tokens_per_minute
    last_update_time = time.time()

    # initialize flags
    file_not_finished = True  # after file is empty, we'll skip reading it
    logging.debug("Initialization complete.")

    requests = self.input_prompts_list.__iter__()

    logging.debug("File opened. Entering main loop")

    task_list = []

    while True:
      # get next request (if one is not already waiting for capacity)
      if next_request is None:
        if not queue_of_requests_to_retry.empty():
          next_request = queue_of_requests_to_retry.get_nowait()
          logging.debug(f"Retrying request {next_request.task_id}: {next_request}")
        elif file_not_finished:
          try:
            # get new request
            # request_json = json.loads(next(requests))
            request_json = next(requests)

            next_request = APIRequest(task_id=next(task_id_generator),
                                      request_json=request_json,
                                      token_consumption=num_tokens_consumed_from_request(
                                          request_json, api_endpoint, self.token_encoding_name),
                                      attempts_left=self.max_attempts,
                                      metadata=request_json.pop("metadata", None))
            status_tracker.num_tasks_started += 1
            status_tracker.num_tasks_in_progress += 1
            logging.debug(f"Reading request {next_request.task_id}: {next_request}")
          except StopIteration:
            # if file runs out, set flag to stop reading it
            logging.debug("Read file exhausted")
            file_not_finished = False

      # update available capacity
      current_time = time.time()
      seconds_since_update = current_time - last_update_time
      available_request_capacity = min(
          available_request_capacity + self.max_requests_per_minute * seconds_since_update / 60.0,
          self.max_requests_per_minute,
      )
      available_token_capacity = min(
          available_token_capacity + self.max_tokens_per_minute * seconds_since_update / 60.0,
          self.max_tokens_per_minute,
      )
      last_update_time = current_time

      # if enough capacity available, call API
      if next_request:
        next_request_tokens = next_request.token_consumption
        if (available_request_capacity >= 1 and available_token_capacity >= next_request_tokens):
          # update counters
          available_request_capacity -= 1
          available_token_capacity -= next_request_tokens
          next_request.attempts_left -= 1

          # call API
          # TODO: NOT SURE RESPONSE WILL WORK HERE
          task = asyncio.create_task(
              next_request.call_api(
                  request_url=self.request_url,
                  request_header=request_header,
                  retry_queue=queue_of_requests_to_retry,
                  status_tracker=status_tracker,
              ))
          task_list.append(task)
          next_request = None  # reset next_request to empty

          # print("status_tracker.num_tasks_in_progress", status_tracker.num_tasks_in_progress)
          # one_task_result = task.result()
          # print("one_task_result", one_task_result)

      # if all tasks are finished, break
      if status_tracker.num_tasks_in_progress == 0:
        break

      # main loop sleeps briefly so concurrent tasks can run
      await asyncio.sleep(seconds_to_sleep_each_loop)

      # if a rate limit error was hit recently, pause to cool down
      seconds_since_rate_limit_error = (time.time() - status_tracker.time_of_last_rate_limit_error)
      if seconds_since_rate_limit_error < seconds_to_pause_after_rate_limit_error:
        remaining_seconds_to_pause = (seconds_to_pause_after_rate_limit_error - seconds_since_rate_limit_error)
        await asyncio.sleep(remaining_seconds_to_pause)
        # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
        logging.warn(
            f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}"
        )

    # after finishing, log final status
    logging.info("""Parallel processing complete. About to return.""")
    if status_tracker.num_tasks_failed > 0:
      logging.warning(f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed.")
    if status_tracker.num_rate_limit_errors > 0:
      logging.warning(
          f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate.")

    # asyncio wait for task_list
    await asyncio.wait(task_list)

    for task in task_list:
      openai_completion = task.result()
      self.results.append(openai_completion)

    self.cleaned_results: List[str] = extract_context_from_results(self.results)


def extract_context_from_results(results: List[Any]) -> List[str]:
  assistant_contents = []
  total_prompt_tokens = 0
  total_completion_tokens = 0

  for element in results:
    if element is not None:
      for item in element:
        if 'choices' in item:
          for choice in item['choices']:
            if choice['message']['role'] == 'assistant':
              assistant_contents.append(choice['message']['content'])
              total_prompt_tokens += item['usage']['prompt_tokens']
              total_completion_tokens += item['usage']['completion_tokens']
  # Note: I don't think the prompt_tokens or completion_tokens is working quite right...

  return assistant_contents


# dataclasses


@dataclass
class StatusTracker:
  """Stores metadata about the script's progress. Only one instance is created."""

  num_tasks_started: int = 0
  num_tasks_in_progress: int = 0  # script ends when this reaches 0
  num_tasks_succeeded: int = 0
  num_tasks_failed: int = 0
  num_rate_limit_errors: int = 0
  num_api_errors: int = 0  # excluding rate limit errors, counted above
  num_other_errors: int = 0
  time_of_last_rate_limit_error: float = 0  # used to cool off after hitting rate limits


@dataclass
class APIRequest:
  """Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""

  task_id: int
  request_json: dict
  token_consumption: int
  attempts_left: int
  metadata: dict
  result: list = field(default_factory=list)

  async def call_api(
      self,
      request_url: str,
      request_header: dict,
      retry_queue: asyncio.Queue,
      status_tracker: StatusTracker,
  ):
    """Calls the OpenAI API and saves results."""
    # logging.info(f"Starting request #{self.task_id}")
    error = None
    try:
      async with aiohttp.ClientSession() as session:
        async with session.post(url=request_url, headers=request_header, json=self.request_json) as response:
          response = await response.json()
      if "error" in response:
        logging.warning(f"Request {self.task_id} failed with error {response['error']}")
        status_tracker.num_api_errors += 1
        error = response
        if "Rate limit" in response["error"].get("message", ""):
          status_tracker.time_of_last_rate_limit_error = time.time()
          status_tracker.num_rate_limit_errors += 1
          status_tracker.num_api_errors -= 1  # rate limit errors are counted separately

    except Exception as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
      logging.warning(f"Request {self.task_id} failed with Exception {e}")
      status_tracker.num_other_errors += 1
      error = e
    if error:
      self.result.append(error)
      if self.attempts_left:
        retry_queue.put_nowait(self)
      else:
        logging.error(f"Request {self.request_json} failed after all attempts. Saving errors: {self.result}")
        data = ([self.request_json, [str(e) for e in self.result], self.metadata]
                if self.metadata else [self.request_json, [str(e) for e in self.result]])
        #append_to_jsonl(data, save_filepath)
        status_tracker.num_tasks_in_progress -= 1
        status_tracker.num_tasks_failed += 1
        return data
    else:
      data = ([self.request_json, response, self.metadata] if self.metadata else [self.request_json, response]
             )  # type: ignore
      #append_to_jsonl(data, save_filepath)
      status_tracker.num_tasks_in_progress -= 1
      status_tracker.num_tasks_succeeded += 1
      # logging.debug(f"Request {self.task_id} saved to {save_filepath}")

      return data


# functions


def api_endpoint_from_url(request_url: str):
  """Extract the API endpoint from the request URL."""
  if 'text-embedding-ada-002' in request_url:
    return 'embeddings'
  else:
    match = re.search('^https://[^/]+/v\\d+/(.+)$', request_url)
    return match[1]  # type: ignore


def append_to_jsonl(data, filename: str) -> None:
  """Append a json payload to the end of a jsonl file."""
  json_string = json.dumps(data)
  with open(filename, "a") as f:
    f.write(json_string + "\n")


def num_tokens_consumed_from_request(
    request_json: dict,
    api_endpoint: str,
    token_encoding_name: str,
):
  """Count the number of tokens in the request. Only supports completion and embedding requests."""
  encoding = tiktoken.get_encoding(token_encoding_name)
  # if completions request, tokens = prompt + n * max_tokens
  if api_endpoint.endswith("completions"):
    max_tokens = request_json.get("max_tokens", 15)
    n = request_json.get("n", 1)
    completion_tokens = n * max_tokens

    # chat completions
    if api_endpoint.startswith("chat/"):
      num_tokens = 0
      for message in request_json["messages"]:
        num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        for key, value in message.items():
          num_tokens += len(encoding.encode(value))
          if key == "name":  # if there's a name, the role is omitted
            num_tokens -= 1  # role is always required and always 1 token
      num_tokens += 2  # every reply is primed with <im_start>assistant
      return num_tokens + completion_tokens
    # normal completions
    else:
      prompt = request_json["prompt"]
      if isinstance(prompt, str):  # single prompt
        prompt_tokens = len(encoding.encode(prompt))
        num_tokens = prompt_tokens + completion_tokens
        return num_tokens
      elif isinstance(prompt, list):  # multiple prompts
        prompt_tokens = sum([len(encoding.encode(p)) for p in prompt])
        num_tokens = prompt_tokens + completion_tokens * len(prompt)
        return num_tokens
      else:
        raise TypeError('Expecting either string or list of strings for "prompt" field in completion request')
  # if embeddings request, tokens = input tokens
  elif api_endpoint == "embeddings":
    input = request_json["input"]
    if isinstance(input, str):  # single input
      num_tokens = len(encoding.encode(input))
      return num_tokens
    elif isinstance(input, list):  # multiple inputs
      num_tokens = sum([len(encoding.encode(i)) for i in input])
      return num_tokens
    else:
      raise TypeError('Expecting either string or list of strings for "inputs" field in embedding request')
  # more logic needed to support other API calls (e.g., edits, inserts, DALL-E)
  else:
    raise NotImplementedError(f'API endpoint "{api_endpoint}" not implemented in this script')


def task_id_generator_function():
  """Generate integers 0, 1, 2, and so on."""
  task_id = 0
  while True:
    yield task_id
    task_id += 1


if __name__ == '__main__':
  pass

  # run script
  # if __name__ == "__main__":
  #   qdrant_client = QdrantClient(
  #     url=os.getenv('QDRANT_URL'),
  #     api_key=os.getenv('QDRANT_API_KEY'),
  #   )
  #   vectorstore = Qdrant(
  #         client=qdrant_client,
  #         collection_name=os.getenv('QDRANT_COLLECTION_NAME'),  # type: ignore
  #         embeddings=OpenAIEmbeddings())  # type: ignore

  #   user_question = "What is the significance of Six Sigma?"
  #   k = 4
  #   fetch_k = 200
  #   found_docs = vectorstore.max_marginal_relevance_search(user_question, k=k, fetch_k=200)

  #   requests = []
  #   for i, doc in enumerate(found_docs):
  #     dictionary = {
  #         "model": "gpt-3.5-turbo-0613", # 4k context
  #         "messages": [{
  #             "role": "system",
  #             "content": "You are a factual summarizer of partial documents. Stick to the facts (including partial info when necessary to avoid making up potentially incorrect details), and say I don't know when necessary."
  #         }, {
  #             "role":
  #                 "user",
  #             "content":
  #                 f"What is a comprehensive summary of the given text, based on the question:\n{doc.page_content}\nQuestion: {user_question}\nThe summary should cover all the key points only relevant to the question, while also condensing the information into a concise and easy-to-understand format. Please ensure that the summary includes relevant details and examples that support the main ideas, while avoiding any unnecessary information or repetition. Feel free to include references, sentence fragments, keywords, or anything that could help someone learn about it, only as it relates to the given question. The length of the summary should be as short as possible, without losing relevant information.\n"
  #         }],
  #         "n": 1,
  #         "max_tokens": 500,
  #         "metadata": doc.metadata
  #     }
  #     requests.append(dictionary)

  #   oai = OpenAIAPIProcessor(
  #       input_prompts_list=requests,
  #       request_url='https://api.openai.com/v1/chat/completions',
  #       api_key=os.getenv("OPENAI_API_KEY"),
  #       max_requests_per_minute=1500,
  #       max_tokens_per_minute=90000,
  #       token_encoding_name='cl100k_base',
  #       max_attempts=5,
  #       logging_level=20,
  #   )
  #   # run script
  #   asyncio.run(oai.process_api_requests_from_file())

  #   assistant_contents = []
  #   total_prompt_tokens = 0
  #   total_completion_tokens = 0

  #   print("Results, end of main: ", oai.results)
  #   print("-"*50)

  #   # jsonObject = json.loads(oai.results)
  #   for element in oai.results:
  #       for item in element:
  #           if 'choices' in item:
  #               for choice in item['choices']:
  #                   if choice['message']['role'] == 'assistant':
  #                       assistant_contents.append(choice['message']['content'])
  #               total_prompt_tokens += item['usage']['prompt_tokens']
  #               total_completion_tokens += item['usage']['completion_tokens']

  #   print("Assistant Contents:", assistant_contents)
  #   print("Total Prompt Tokens:", total_prompt_tokens)
  #   print("Total Completion Tokens:", total_completion_tokens)
  #   turbo_total_cost = (total_prompt_tokens * 0.0015) + (total_completion_tokens * 0.002)
  #   print("Total cost (3.5-turbo):", (total_prompt_tokens * 0.0015), " + Completions: ", (total_completion_tokens * 0.002), " = ", turbo_total_cost)

  #   gpt4_total_cost = (total_prompt_tokens * 0.03) + (total_completion_tokens * 0.06)
  #   print("Hypothetical cost for GPT-4:", (total_prompt_tokens * 0.03), " + Completions: ", (total_completion_tokens * 0.06), " = ", gpt4_total_cost)
  #   print("GPT-4 cost premium: ", (gpt4_total_cost / turbo_total_cost), "x")
  '''
  Pricing:
  GPT4: 
    * $0.03 prompt
    * $0.06 completions
  3.5-turbo: 
    * $0.0015 prompt
    * $0.002 completions
  '''
"""
APPENDIX

The example requests file at openai-cookbook/examples/data/example_requests_to_parallel_process.jsonl contains 10,000 requests to text-embedding-ada-002.

It was generated with the following code:

```python
import json

filename = "data/example_requests_to_parallel_process.jsonl"
n_requests = 10_000
jobs = [{"model": "text-embedding-ada-002", "input": str(x) + "\n"} for x in range(n_requests)]
with open(filename, "w") as f:
    for job in jobs:
        json_string = json.dumps(job)
        f.write(json_string + "\n")
```

As with all jsonl files, take care that newlines in the content are properly escaped (json.dumps does this automatically).
"""
