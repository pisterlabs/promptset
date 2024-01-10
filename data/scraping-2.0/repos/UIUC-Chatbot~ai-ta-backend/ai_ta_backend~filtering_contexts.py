import json
import os
import time
from typing import Optional

import openai
import ray
import requests
# from langchain import hub
# import replicate
from posthog import Posthog
import sentry_sdk

# from dotenv import load_dotenv
# load_dotenv(override=True)
# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")

filter_unrelated_contexts_zephyr = """<|system|>
You are an expert at determining if a passage is relevant and helpful for answering a question.
To be valuable, a passage must have at least some amount of useful and meaningful information with more than a passing mention of the topic.
As part of your thinking process, you first write a few sentences evaluating the utility of the passage, given the question we're trying to answer. Limit yourself to writing only a sentence or two, no more.
Finally, you must submit your final answer by adding two newline characters then "Yes." or "No." or "I don't know.". Provide a single answer only. Providing multiple final results will disqualify you.
Here's a template code snippet of how it should work (with placeholder variables):
```
Passage: <The full text of the passage>
Question: <The question we're using to determine relevancy of the passage>
Your evaluation of the utility of the passage: <A few sentences exploring how useful the passage is for this particular question>


Final answer: <Enter one of "Yes." or "No." or "I don't know."
```
Here's a complete example. Follow this formatting exactly.
```
Passage: Figure 4.6: Overview of the CUDA device memory model 
 
In order to fully appreciate the difference between registers, shared memory and global memory, we need to go into a little more details of how these different memory types are realized and used in modern processors. Virtually all modern processors find their root in the model proposed by John von Neumann in 1945, which is shown in Figure 4.7. The CUDA devices are no exception. The Global Memory in a CUDA device maps to the Memory box in Figure 4.7. The processor box corresponds to the processor chip boundary that we typically see today. The Global Memory is off the processor chip and is implemented with DRAM technology, which implies long access latencies and relatively low access bandwidth.

Question: Explain how tiling helps with global memory bandwidth.
Your evaluation of the utility of the passage: The passage briefly mentions the use of shared memory as a means to reduce global memory bandwidth, but it doesn't provide a detailed explanation or analysis of how tiling helps with global memory bandwidth. Therefore, the passage is not helpful when answering the question.


Final answer: No.
```</s>
<|user|>
Passage: {context}
Question: {user_query}
Your evaluation of the utility of the passage: </s>
<|assistant|>"""


@ray.remote
class AsyncActor:

  def filter_context(self, context, user_query, langsmith_prompt_obj):
    final_prompt = str(langsmith_prompt_obj.format(context=context, user_query=user_query))
    # print(f"-------\nfinal_prompt:\n{final_prompt}\n^^^^^^^^^^^^^")
    try:
      # completion = run_caii_hosted_llm(final_prompt)
      # completion = run_replicate(final_prompt)
      completion = run_anyscale(final_prompt)
      return {"completion": completion, "context": context}
    except Exception as e:
      sentry_sdk.capture_exception(e)
      print(f"Error: {e}")


def run_caii_hosted_llm(prompt, max_tokens=300, temp=0.3, **kwargs):
  """
  Local LLMs  USAGE DOCS: https://kastanday.notion.site/LLM-Serving-on-prem-OpenAI-Clone-bb06028266d842b0872465f552684177 ## 
  """

  url = "http://api.kastan.ai/v1/completions?model=HuggingFaceH4/zephyr-7b-alpha"
  headers = {'Content-Type': 'application/json'}
  data = {"prompt": prompt, "max_tokens": max_tokens, "temperature": temp, **kwargs}

  response = None
  try:
    response = requests.post(url, headers=headers, data=json.dumps(data), timeout=180)
    return response.json()['choices'][0]['text']
  except Exception as e:
    sentry_sdk.capture_exception(e)
    # Probably cuda OOM error.
    response_content = response.json() if response else "No response"
    raise ValueError(
        f"🚫🚫🚫 Failed inference attempt. Response: {response_content}\nError: {e}\nPromt that caused error: {prompt}"
    ) from e


def run_replicate(prompt):
  output = None
  # output = replicate.run("tomasmcm/zephyr-7b-beta:961cd6665b811d0c43c0b9488b6dfa85ff5c7bfb875e93b4533e4c7f96c7c526",
  #                        input={
  #                            "top_k": 50,
  #                            "top_p": 0.95,
  #                            "prompt": prompt,
  #                            "temperature": 0.3,
  #                            "max_new_tokens": 250,
  #                            "presence_penalty": 1
  #                        })
  print(output)
  return output


def run_anyscale(prompt, model_name="HuggingFaceH4/zephyr-7b-beta"):
  start_time = time.monotonic()
  ret = openai.ChatCompletion.create(
      api_base="https://api.endpoints.anyscale.com/v1",
      api_key=os.environ["ANYSCALE_ENDPOINT_TOKEN"],
      api_type="openai",
      # model="mistralai/Mistral-7B-Instruct-v0.1",
      model="HuggingFaceH4/zephyr-7b-beta",
      messages=[{
          "role": "system",
          "content": "You are a helpful assistant."
      }, {
          "role": "user",
          "content": prompt
      }],
      temperature=0.3,
      max_tokens=250,
  )

  output = ret["choices"][0]["message"]["content"]  # type: ignore
  print("Response from Anyscale:", output[:150])

  # input_length = len(tokenizer.encode(prompt))
  # output_length = len(tokenizer.encode(output))
  # Input tokens {input_length}, output tokens: {output_length}"
  print(f"^^^^ one anyscale call Runtime: {(time.monotonic() - start_time):.2f} seconds.")
  return output


def parse_result(result: str):
  lines = result.split('\n')
  for line in lines:
    if 'Final answer' in line:
      return 'yes' in line.lower()
  return False


def filter_top_contexts(contexts,
                        user_query: str,
                        timeout: Optional[float] = None,
                        max_concurrency: Optional[int] = 180):

  print("⏰⏰⏰ Starting filter_top_contexts() ⏰⏰⏰")

  timeout = timeout or float(os.environ["FILTER_TOP_CONTEXTS_TIMEOUT_SECONDS"])
  # langsmith_prompt_obj = hub.pull("kastanday/filter-unrelated-contexts-zephyr") # TOO UNSTABLE, service offline
  langsmith_prompt_obj = filter_unrelated_contexts_zephyr
  posthog = Posthog(project_api_key=os.environ['POSTHOG_API_KEY'], host='https://app.posthog.com')

  print("Max concurrency:", max_concurrency)
  print("Num contexts to filter:", len(contexts))

  # START TASKS
  actor = AsyncActor.options(max_concurrency=max_concurrency, num_cpus=0.001).remote()  # type: ignore
  result_futures = [actor.filter_context.remote(c, user_query, langsmith_prompt_obj) for c in contexts]

  start_time = time.monotonic()
  done_tasks, in_progress = ray.wait(result_futures,
                                     num_returns=len(result_futures),
                                     timeout=timeout,
                                     fetch_local=False)

  # Cleanup
  for task in in_progress:
    ray.cancel(task)
  results = ray.get(done_tasks)
  ray.kill(actor)

  best_contexts_to_keep = [
      r['context'] for r in results if r and 'context' in r and 'completion' in r and parse_result(r['completion'])
  ]

  print("🧠🧠 TOTAL DOCS PROCESSED BY ANYSCALE FILTERING:", len(results))
  print("🧠🧠 TOTAL DOCS KEPT, AFTER FILTERING:", len(best_contexts_to_keep))
  mqr_runtime = round(time.monotonic() - start_time, 2)
  print(f"⏰ Total elapsed time: {mqr_runtime} seconds")

  posthog.capture('distinct_id_of_the_user',
                  event='filter_top_contexts',
                  properties={
                      'user_query': user_query,
                      'course_name': contexts[0].metadata.get('course_name', None),
                      'percent_kept': len(best_contexts_to_keep) / len(results),
                      'total_docs_processed': len(results),
                      'total_docs_kept': len(best_contexts_to_keep),
                      'MQR_total_runtime_sec': mqr_runtime,
                  })
  return best_contexts_to_keep


def run_main():
  start_time = time.monotonic()
  # final_passage_list = filter_top_contexts(contexts=CONTEXTS * 2, user_query=USER_QUERY)
  # print("✅✅✅ TOTAL included in results: ", len(final_passage_list))
  print(f"⏰⏰⏰ Runtime: {(time.monotonic() - start_time):.2f} seconds")
  # print("Total contexts:", len(CONTEXTS) * 2)


# ! CONDA ENV: llm-serving
if __name__ == "__main__":
  run_main()
