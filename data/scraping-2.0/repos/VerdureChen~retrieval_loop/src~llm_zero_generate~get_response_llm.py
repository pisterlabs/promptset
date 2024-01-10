import argparse
import os
import json
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm
import openai
import os
import datasets
import sys
sys.path.append('../retrieval_loop')
from elastic_bm25_search_with_metadata import ElasticSearchBM25Retriever
from eva_generate import evaluate

Mode = {
    "Zero-shot": 0,
    "With-context": 0,
}

Prompt = {
    "Zero-shot": "Provide a background document in 100 words according to your knowledge from Wikipedia to answer the given question."
                 "\n\nQuestion:{question} \n\nBackground Document:",
    "With-context": "Context information is below.\n"
                  "---------------------\n"
                  "{context_str}\n"
                  "---------------------\n"
                  "Using both the context information and also using your own knowledge, answer the following question"
                  "with a background document in 100 words.\n\nQuestion:{question} \n\nBackground Document:",
}


def get_openai_api(model_name):
    if model_name == "Qwen":
        print("Using Qwen")
        openai.api_base = "http://0.0.0.0:8111/v1"
        openai.api_key = "xxx"
    elif model_name == "Llama":
        print("Using Llama")
        openai.api_base = "http://0.0.0.0:8223/v1"
        openai.api_key = "xxx"
    elif model_name == "chatglm3":
        print("Using chatglm3")
        openai.api_base = "http://0.0.0.0:8113/v1"
        openai.api_key = "xxx"
    elif model_name == "baichuan2-13b-chat":
        print("Using baichuan2-13b-chat")
        openai.api_base = "http://0.0.0.0:8222/v1"
        openai.api_key = "xxx"
    elif model_name == "gpt-3.5-turbo":
        print("Using gpt-3.5-turbo")
        # openai.api_base = "https://one-api.ponte.top/v1"
        # openai.api_key = "sk-9y7d4TtOJO3xzHXn6dCa9fE02d18436d86Be540a00DcD1F8"
        openai.api_base = "http://47.245.109.131:5555/v1"
        openai.api_key = "sk-ZW18DSR8JBsNKOg2Af61E55bD23c481dAa916f624e8c9d65"
    else:
        raise ValueError("Model name not supported")

def get_args():
    # get config_file_path, which is the path to the config file
    # config file formatted as a json file:
    # {
    #   "model_name": "Qwen",
    #   "question_file_path": "data/qa/qa.json",
    #   "output_file_path": "data/qa/qa_with_response.json"
    # }
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file_path", type=str, required=True)
    args = parser.parse_args()
    # read config file
    config_file_path = args.config_file_path
    with open(config_file_path, "r") as f:
        config = json.load(f)
    return config


@retry(stop=stop_after_attempt(20), wait=wait_exponential(multiplier=1, max=10))
def get_response_llm(model_name, text, filter_words=None):
    completion = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "user",
             "content": text},
        ],
        max_tokens=128,
        temperature=0.7,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0.6,
        stop=filter_words,
        stream=False,
    )


    # print(text)
    # print(completion.choices[0].message.content)
    # print("\n")

    resp = completion.choices[0].message.content.strip().replace("\n", " ")
    # memory cleanup
    del completion
    return resp


def prepare_input(question, context, index, mode, context_ref_num=0):
    if mode == Mode["Zero-shot"]:
        prompt = Prompt["Zero-shot"]
        prompt = prompt.format(question=question)
    elif mode == Mode["With-context"]:
        prompt = Prompt["With-context"]
        # get context text
        contexts = context[:context_ref_num]
        context_doc_ids = [item["docid"] for item in contexts]
        context_texts = [index.get_document_by_id(doc_id).page_content for doc_id in context_doc_ids]
        context_str = "\n".join(f"[Context {i+1}]: {context_text}" for i, context_text in enumerate(context_texts))
        prompt = prompt.format(question=question, context_str=context_str)
    else:
        raise ValueError("Mode not supported")
    print(prompt)
    return prompt


def get_index(elasticsearch_url, index_name, with_context=False):
    if with_context:
        index = ElasticSearchBM25Retriever.create(elasticsearch_url, index_name, verbose=False)
    else:
        index = None
    return index


def read_dataset(question_file_path, with_context):
    if with_context:
        print("Using context")
        Mode["With-context"] = 1
        with open(question_file_path, "r", encoding='utf-8') as f:
            question_dataset = json.load(f)
        formatted_data = [
            {
                "id": qid,
                "question": details["question"],
                "answers": details["answers"],
                "contexts": details["contexts"],
            }
            for qid, details in question_dataset.items()
        ]
        question_dataset = datasets.Dataset.from_list(formatted_data)

    else:
        print("Not using context")
        Mode["Zero-shot"] = 1
        question_dataset = datasets.load_dataset("json", data_files=question_file_path)["train"]


    return question_dataset


def get_response(example, with_context=0, context_ref_num=0, elasticsearch_url=None, index_name=None, model_name=None):
    index = get_index(elasticsearch_url, index_name, with_context)
    question = example["question"]
    if "contexts" in example:
        context = example["contexts"]
    else:
        context = ""
    input_text = prepare_input(question, context, index=index, mode=with_context, context_ref_num=context_ref_num)
    filter_words = [" Human:", " AI:", "sorry", " Sorry"]
    # if question contains filter_words, remove the word from filter_words
    for filter_word in filter_words:
        if filter_word.strip().lower() in input_text.lower():
            # remove all instances of filter_words
            filter_words = [word for word in filter_words if word != filter_word]

    response = ""
    while response.strip() == "" \
            or len(response.split(" ")) < 5 \
            or [word for word in filter_words if word in response] != []:
        response = get_response_llm(model_name, input_text, filter_words=filter_words)
    # if response > 100 words, truncate to 100 words
    response = " ".join(response.split(" ")[:100])
    example["response"] = response
    return example


if __name__ == '__main__':
    config = get_args()
    model_name = config["model_name"]
    question_file_path = config["question_file_path"]
    output_file_path = config["output_file_path"]
    with_context = config["with_context"]
    elastic_url = config["elasticsearch_url"]
    index_name = config["index_name"]
    context_ref_num = int(config["context_ref_num"])
    print(config)
    # config_file:
    # {
    #   "model_name": "Qwen",
    #   "question_file_path": "data/qa/qa.json",
    #   "output_file_path": "data/qa/qa_with_response.json"
    #   "with_context": 1,
    #   "elasticsearch_url": "http://localhost:9200",
    #   "index_name": "wiki",
    #   "context_ref_num": 3
    # }

    # set openai api
    get_openai_api(model_name)
    # read dataset
    question_dataset = read_dataset(question_file_path, with_context)
    # set kwargs
    map_fn_kwargs = {"with_context": Mode["With-context"], "context_ref_num": context_ref_num,
                     'elasticsearch_url': elastic_url, 'index_name': index_name, 'model_name': model_name}
    # check mode
    assert Mode["Zero-shot"] + Mode["With-context"] == 1, "Only one mode can be used at a time"
    print(f"Number of examples: {len(question_dataset)}")



    # question_dataset = question_dataset.map(get_response, num_proc=4, fn_kwargs=map_fn_kwargs)
    # question_dataset.to_json(output_file_path, force_ascii=False)

    shard_size = 500
    num_shards = len(question_dataset) // shard_size
    if num_shards == 0:
        num_shards = 1

    shard_names = [f"{output_file_path}_shard_{i}.json" for i in range(num_shards)]
    existing_shards = [shard_name for shard_name in shard_names if os.path.exists(shard_name)]
    print(f"Existing shards: {existing_shards}")
    print(f"Shard names: {shard_names}")

    for shard_name in shard_names:
        if shard_name not in existing_shards:
            print('-' * 50)
            print(f"Creating shard {shard_name}")
            print(f"Shard index: {shard_names.index(shard_name)}")
            print('-' * 50)
            shard_dataset = question_dataset.shard(num_shards, index=shard_names.index(shard_name))
            shard_dataset = shard_dataset.map(get_response, num_proc=8, fn_kwargs=map_fn_kwargs)
            # only keep 'id', 'question', 'answers', 'response'
            shard_dataset = shard_dataset.select_columns(['id', 'question', 'answers', 'response'])
            shard_dataset.to_json(shard_name, force_ascii=False)
        else:
            print('-' * 50)
            print(f"Skipping shard {shard_name} of index {shard_names.index(shard_name)}")

    # combine shards
    print('-' * 50)
    print("Combining shards")
    print('-' * 50)
    combined_dataset = datasets.load_dataset("json", data_files=shard_names)["train"]
    # 先使用map方法创建一个临时的整数类型的id字段，例如命名为"id_int"
    combined_dataset = combined_dataset.map(lambda example: {"id_int": int(example["id"])})

    # 根据这个新字段排序
    combined_dataset = combined_dataset.sort("id_int")

    # 删除这个新字段
    combined_dataset = combined_dataset.remove_columns(["id_int"])

    # evaluate
    print('-' * 50)
    print("Evaluating")
    print('-' * 50)
    prediction = evaluate(combined_dataset)
    EM = sum(prediction['exact_match']) / len(prediction['exact_match'])
    print(f"EM:{EM}")

    #write metrics to file
    metrics_file_path = os.path.join(os.path.dirname(output_file_path), f"{model_name}_rag_metrics.json")
    with open(metrics_file_path, "a") as f:
        f.write(f"{output_file_path}_{context_ref_num}: {EM}\n")


    combined_dataset.to_json(output_file_path, force_ascii=False)

    # delete shards
    print('-' * 50)
    print("Deleting shards")
    print('-' * 50)
    for shard_name in shard_names:
        os.remove(shard_name)
    print("Done")




