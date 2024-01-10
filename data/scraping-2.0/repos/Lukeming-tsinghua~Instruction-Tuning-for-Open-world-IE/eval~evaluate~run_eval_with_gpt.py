import openai
import json
import copy
from multiprocess import Pool
from tqdm import tqdm
import random


openai.api_key = "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"


raw_instruction = """
Please provide the response in the JSON format. The response should contains entities and triplets. Each entity has its mention, title, a list of types, description, and a list of aliases. Each triplet has its head and tail mentions, and a list of relations.
"""

format_instruction = """
Please provide the response in the JSON format. The response should contains entities and triplets. Each entity has its mention, title, a list of types, description, and a list of aliases. Each triplet has its head and tail mentions, and a list of relations. Here is an example of the return JSON format: {"entities": [{"mention": String, "title": String, "type": List[String], "description": String, "aliases":List[String]}], "triplets": [{"head": String, "tail": String, "relations": List[String]}]}.
"""

fewshot_instruction = """Please provide the output of this case and only return the JSON."""

with open("oneshot_history_format.txt") as f:
    oneshot_history = json.load(f)


eval_data = [json.loads(line) for line in open("/harddisk/user/keminglu/pretrained_data_processed/wikipedia_with_mention_wo_title_simplified_aug_eval/corpus_filtered_test_prompt_rephrased_3").readlines()]


def inference(sample, case="oneshot_history"):
  sample = copy.deepcopy(sample)
  if case == "oneshot_history":
    query = f"[input] {sample['inputs']}\n[instruction] {sample['rephrased_prompt']}\n\n" + format_instruction
    try:
      response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=oneshot_history[sample['aug_type']] + [
              {"role": "user", "content": query},
          ]
      )
      message = response['choices'][0]['message']['content']
    except Exception as e:
      print(e)
      message = None
  elif case == "format":
    query = f"[input] {sample['inputs']}\n[instruction] {sample['rephrased_prompt']}\n\n" + format_instruction
    try:
      response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
              {"role": "user", "content": query},
          ]
      )
      message = response['choices'][0]['message']['content']
    except Exception as e:
      message = None

  sample['outputs'] = message
  return sample



results = []
with Pool(32) as p:
    pbar = tqdm(total=len(eval_data))
    for result in p.imap_unordered(inference, eval_data):
        results.append(result)
        pbar.update(1)


with open("/harddisk/user/keminglu/evaluation_corpus/wiki_aug_eval/full_chatgpt_data_v4_corpus_filtered_3_oneshot_format_prompt_case_output.txt", "w") as f:
    for result in results:
        f.write(json.dumps(result) + "\n")
