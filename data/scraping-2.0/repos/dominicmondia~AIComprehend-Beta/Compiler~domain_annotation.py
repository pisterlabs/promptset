import json
import spacy
import openai
import re

# Load the dataset
with open("aicomprehend_annotated_dataset_v5.json", 'r') as f:
    annotation_dataset = json.loads(f.read())

# Load the API key
with open('api.json') as api:
    api_key = json.load(api)
    openai.api_key = api_key['api_key']

# Load the nlp model
nlp = spacy.load("en_core_web_lg")


# Save the dataset locally
def save():
    with open("aicomprehend_annotated_dataset_v6.json", 'w') as f:
        f.write(json.dumps(annotation_dataset, indent=4))


# # Create function to call OpenAI API to get the response from chatgpt 3.5 turbo model
# def bot_annotate_relevant_sentences(data):
#     prompt = f"Extract all the sentences from the passage that are relevant to the question, and choices. Do not answer the question.\n passage: {data['passage']} question: {data['question']} choices: {data['choices']}"
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {"role": "system", "content": "You are a data annotation assistant"},
#             {"role": "user", "content": prompt},
#         ],
#     )
#
#     return response
#
#
# # Annotate the dataset
# for item in annotation_dataset:
#     if "relevant_sentences" in item:
#         continue
#     response = bot_annotate_relevant_sentences(item)
#     item["relevant_sentences"] = response["choices"][0]["message"]["content"].split("\n")
#     save()

# Create function to call OpenAI API to get the response from chatgpt 3.5 turbo model
# def bot_annotate_relevant_sentences(data):
#     prompt = f"Combine the question and each of the choices to form a complete statement for each of the choices. Label each with numbers 1-4" \
#              f" question: {data['question']} " \
#              f"choices: {data['choices']}"
#     response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[
#             {"role": "system", "content": "You are a data annotation assistant"},
#             {"role": "user", "content": prompt},
#         ],
#     )
#
#     return response
#
#
# # annotate the dataset
# for item in annotation_dataset:
#     if "choices_in_complete_thought" in item:
#         continue
#     response = bot_annotate_relevant_sentences(item)
#     item["choices_in_complete_thought"] = response["choices"][0]["message"]["content"].split("\n")
#     save()

for item in annotation_dataset:

    if "best title" in item["question"]:
        item["choices_in_complete_thought"] = []
        for choice in item["choices"]:
            item["choices_in_complete_thought"].append(f"The best title for the passage is {choice}")
    elif "choices_in_complete_thought" not in item:
        item["choices_in_complete_thought"] = item["choices"]
    else:
        # delete empty sentences
        item["choices_in_complete_thought"] = [x for x in item["choices_in_complete_thought"] if x != "" or x != " "]
        for a in range(4):
            item["choices_in_complete_thought"][a] = re.sub(r"\d+\. ", "", item["choices_in_complete_thought"][a])
            item["choices_in_complete_thought"][a] = re.sub(r"\d+\) ", "", item["choices_in_complete_thought"][a])
            if item["choices_in_complete_thought"][a][0] == "-":
                item["choices_in_complete_thought"][a] = item["choices_in_complete_thought"][a][2:]

    # delete empty sentences
    item["relevant_sentences"] = [x for x in item["relevant_sentences"] if x != "" or x != " "]

    # remove the first "- " from each line
    for i in range(len(item["relevant_sentences"])):
        item["relevant_sentences"][i] = re.sub(r"\d+\. ", "", item["relevant_sentences"][i])
        item["relevant_sentences"][i] = re.sub(r"\d+\) ", "", item["relevant_sentences"][i])
        if len(item["relevant_sentences"][i]) > 2:
            if item["relevant_sentences"][i][0] == "-":
                item["relevant_sentences"][i] = item["relevant_sentences"][i][2:]
    save()

