from helpers import get_env
import openai
import pandas as pd
import re
import blobfile as bf
import json
import random
import os
import jsonlines
from typing import List, Tuple
import copy


API_KEY, RESOURCE_ENDPOINT = get_env("azure-openai")

openai.api_type = "azure"
openai.api_key = API_KEY
openai.api_base = RESOURCE_ENDPOINT
openai.api_version = "2022-12-01"

def normalizeText(text: str) -> str:
    text = text.replace("Question:", "")
    text = text.replace("Answer:", "")
    text = re.sub(r"Q\d:", "", text)
    text = re.sub(r"A\d:", "", text)
    text = re.sub(r"Q:", "", text)  
    text = re.sub(r"A:", "", text) 
    text = re.sub(r"\s+", " ", text)
    return text

def manuelly_extract_matches(text: str) -> List[Tuple[str, str]]:
    text = text.replace("?;", "?")
    pairList = text.split(";")
    pairList = [normalizeText(pair) for pair in pairList]
    pairList = [pair.split("?") for pair in pairList]
    pairList = [(pair[0] + "?", pair[1] if len(pair) == 2 else "") for pair in pairList]
    return pairList


def text_to_dictionary(text):
    text = text.replace("\n", " ")
    pattern1 = r"Question: (.*?\?) Answer: (.*?)(?=Question:|$)"
    pattern2 = r"Question: (.*?) Answer: (.*?)(?=Question:|$)"
    matches = re.findall(pattern1, text, re.DOTALL)
    if len(matches) == 0:
        matches = re.findall(pattern2, text, re.DOTALL)
        if len(matches) == 0:
            matches = manuelly_extract_matches(text)
    result = []

    for match in matches:
        question, answer = match
        question = question.strip()
        answer = answer.strip().replace(";", "")
        result.append({"question": question, "answer": answer})

    return result

def edit_jsonl_line(jsonl_file, line_number, edit_func) -> None:
    temp_file = jsonl_file + ".tmp"

    with open(jsonl_file, 'r') as input_file, open(temp_file, 'w') as temp_output:
        for i, line in enumerate(input_file):
            if i == line_number:
                json_data = json.loads(line)
                edited_json_data = edit_func(json_data)
                if len(edited_json_data["qa_pairs"]) == 0:
                    continue
                edited_line = json.dumps(edited_json_data)
                temp_output.write(edited_line + '\n')
            else:
                temp_output.write(line)

    input_file.close()
    temp_output.close()

    os.remove(jsonl_file)
    os.rename(temp_file, jsonl_file)

def print_category_stats(category: str) -> None:
    with jsonlines.open(f"all_dataset.jsonl") as reader:
        dataset = list(reader)
    dataset = [entry for entry in dataset if entry["category"] == category]
    num_questions = sum([len(entry["qa_pairs"]) for entry in dataset])
    num_paragraphs = sum([len(entry["paragraph"]) for entry in dataset])
    num_entries = len(dataset)
    print(f"Category: {category}")
    print(f"Number of entries: {num_entries}")
    print(f"Number of paragraphs: {num_paragraphs}")
    print(f"Number of questions: {num_questions}")

textchunks_df = pd.read_csv("textchunks_df.csv")
textchunks_df = textchunks_df[textchunks_df["text"].notna()]



CATEGORY_DICT = {
    "SAME_WORDING_ONE_PARAGRAPH": "The questions should have the same wording as the given text.",
    "SAME_WORDING_MULTIPLE_PARAGRAPHS": "The questions should be factoid in nature, and use the same wording as the given text, but each requires an understanding of information presented in at least two of the paragraphs to answer. The aim is to assess comprehension of the given content and the interrelation of ideas across the paragraphs.",
    # "DIFFERENT_WORDING_ONE_PARAGRAPH": "The questions should have different wording than the given text.",
    # "DIFFERENT_WORDING_MULTIPLE_PARAGRAPHS": "The questions should be factoid in nature, and use a different wording as the given text, but each requires an understanding of information presented in at least two of the paragraphs to answer. The aim is to assess comprehension of the given content and the interrelation of ideas across the paragraphs.",
}


prompt = """\
You are generating 5 factoid question answer pairs for a given text.
[BEGIN DATA]
************
[Text]: {text}
************
[END DATA]
{category_description}
Please format the answer as a series of questions followed by their corresponding answer, separated by semicolons, in the following format: [Question]: [Answer]; [Question]: [Answer]; [Question]: [Answer]; [Question]: [Answer]; [Question]: [Answer];
"""
q2Create = 50

for category, category_desc in CATEGORY_DICT.items():
    print_category_stats(category)
    run_type = input(f"Would you like to evaluate questions for category {category}? [y/n/c]: ")
    if run_type == "y":
        print(f"Evaluating questions for category {category}...")
        with bf.BlobFile("all_dataset.jsonl", "rb") as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                if data["category"] != category:
                    continue
                if "evaluated" in data and data["evaluated"]:
                    continue
                print("Paragraph:")
                for paragraph in data["paragraph"]:
                    print(paragraph["text"])
                eval_paragraph = input("Evaluate paragraph? [y/n]: ")
                if eval_paragraph == "y":
                    c_data = copy.deepcopy(data)
                    for qa_pair in c_data["qa_pairs"]:
                        print(f"Question: {qa_pair['question']}")
                        print(f"Answer: {qa_pair['answer']}")
                        keep = input("Keep? [y/n]: ")
                        if keep == "n":
                            data["qa_pairs"].remove(qa_pair)
                            edit_jsonl_line("all_dataset.jsonl", i, lambda x: data)
                    data["evaluated"] = True
                    edit_jsonl_line("all_dataset.jsonl", i, lambda x: data)

    elif run_type == "c":
        continue
    else:
        print(f"Generating questions for category {category}...")
        nrCreated = 0
        while nrCreated < q2Create:
            if "ONE_PARAGRAPH" in category:
                sample = textchunks_df.sample()
                sample = sample.drop(columns=["embedding"])
                sample_text = sample["text"].values[0]
            else:
                idx = random.randint(0, 12)
                cluster_counts = textchunks_df.Cluster.value_counts()
                cluster_counts = cluster_counts[cluster_counts > 9]
                cluster_counts = cluster_counts.sort_values(ascending=False)
                sample_df = textchunks_df[textchunks_df.Cluster == cluster_counts.index[idx]]
                num_paragraphs = random.randint(2, 3)
                # sample_df = textchunks_df[textchunks_df["Cluster"] == cluster]
                samples = sample_df.sample(num_paragraphs)
                sample_text = "; ".join([f"\n\nParagraph {i+1}: {text}" for i, text in enumerate(samples["text"].values)])
            formatted_prompt = prompt.format(text=sample_text, category_description=category_desc)
            if input("Do it yourself? [y/n]: ") == "y":
                print(formatted_prompt)
                result = input("Enter your questions and answers: ")
            else:
                result = openai.Completion.create(
                    engine="davinci",
                    prompt=formatted_prompt,
                    temperature=0.9,
                    max_tokens=1000,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )["choices"][0]["text"]
            qa_pairs = text_to_dictionary(result)
            with bf.BlobFile("all_dataset.jsonl", "ab") as f:
                f.write(
                    (json.dumps(
                        {
                            "category": category,
                            "paragraph": sample.to_dict(orient='records') if "ONE_PARAGRAPH" in category else samples.to_dict(orient='records'),
                            "qa_pairs": qa_pairs,
                        }
                    )
                    + "\n").encode("utf-8")
                )
            nrCreated += 5






