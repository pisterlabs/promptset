import csv
import difflib

import django
django.setup()
import typer
import json
from sefaria.model import *
from sefaria.utils.hebrew import strip_cantillation
import random
import os
from langchain.chat_models import ChatOpenAI
import openai
api_key = os.getenv("OPENAI_API_KEY")


seed_value = 613
random.seed(seed_value)

# def create_data(output_training_filename: str, output_validation_filename: str):
#     all_samples = []
#     for masechet in masechtot_ordered:
#         print("creating data from Masechet " + masechet)
#         all_segment_refs = Ref(masechet).all_segment_refs()
#         for segment_ref in all_segment_refs:
#             non_punctuated = segment_ref.text('he', "William Davidson Edition - Aramaic").text
#             punctuated = strip_cantillation(segment_ref.text('he').text, strip_vowels=True)
#             steinsalz = Ref("Steinsaltz on " + segment_ref.normal()).text('he').text
#             all_samples.append(create_new_context(task_desciption, non_punctuated, steinsalz, punctuated))
#         if masechet == last_masechet:
#             break
#
#     #get only limited num of samples
#     samples_trimmed = []
#     samples_trimmed = random.sample(all_samples, sample_size)
#
#     # Calculate the number of items for training
#     num_train = int(len(samples_trimmed) * train_proportion)
#
#     # Use random.sample to partition the list according to the seed
#     train_samples = random.sample(samples_trimmed, num_train)
#     validation_samples = [item for item in samples_trimmed if item not in train_samples]
#
#     with open(output_training_filename, 'w', encoding='utf-8') as jsonl_file:
#         for json_obj in train_samples:
#             # Use ensure_ascii=False to encode Unicode characters
#             json_line = json.dumps(json_obj, ensure_ascii=False)
#             jsonl_file.write(json_line + '\n')
#     with open(output_validation_filename, 'w', encoding='utf-8') as jsonl_file:
#         for json_obj in validation_samples:
#             # Use ensure_ascii=False to encode Unicode characters
#             json_line = json.dumps(json_obj, ensure_ascii=False)
#             jsonl_file.write(json_line + '\n')
#
#
#     print("TRAINING SAMPLES: "  + str(len(train_samples)))
#     print("VALIDATION SAMPLES: " + str(len(validation_samples)))
def write_lists_to_csv(list1, list2, filename, header1, header2):
    # Combine the lists into a list of tuples
    data = list(zip(list1, list2))

    # Open the CSV file in write mode
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        # Create a CSV writer
        csvwriter = csv.writer(csvfile)

        # Write the headers
        csvwriter.writerow([header1, header2])

        # Write the data
        csvwriter.writerows(data)
def read_json_lines_to_list(file_path):
    data_list = []

    with open(file_path, 'r', encoding='utf-8') as json_file:
        for line in json_file:
            try:
                data = json.loads(line)
                data_list.append(data)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")

    return data_list

    return data_list

def get_response_openai(sample, model_name):
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": sample["messages"][0]["content"]
            },
            {
                "role": "user",
                "content": sample["messages"][1]["content"]
            }

        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    # print(response)
    inference = response["choices"][0]["message"]["content"]
    return(inference)


    print()  # Move to the next line
if __name__ == '__main__':
    # typer.run(visualize)
    print("hi")
    model_name = "ft:gpt-3.5-turbo-0613:sefaria:he-punct:8ClpgehI"
    golden_standard = read_json_lines_to_list('../output/gpt_punctuation_validation.jsonl')
    golden_standard = random.sample(golden_standard, 50)
    inferred = []
    for sample in golden_standard:
        inferred.append(get_response_openai(sample, model_name))
    golden_standard_valids = [sample["messages"][2]["content"] for sample in golden_standard]
    # golden_standard_valids_steinsaltz = [sample["messages"][1]["content"].split('"steinsaltz":')[1][:-1] for sample in golden_standard]
    write_lists_to_csv(golden_standard_valids, inferred, '../output/discrepancies_visualization.csv', "Gold", "Inferred")





