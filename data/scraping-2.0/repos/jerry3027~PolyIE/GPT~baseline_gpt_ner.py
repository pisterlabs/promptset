import re
import json
import time
import openai
import asyncio
from tqdm import tqdm
from GPT import prompt_examples

# Common API key
openai.api_key = "***"

# Azure API Key
openai.api_type = "azure"
openai.api_base = "***"
openai.api_version = "2023-07-01-preview"
openai.api_key = "***"

def data_preprocessor():
    with open('./Cleaned_data/Final_v2/train_split.txt', 'r') as train_file:
        train_dataset = json.load(train_file)
    with open('./Cleaned_data/Final_v2/validation_split.txt', 'r') as validation_file:
        validation_dataset = json.load(validation_file)
    with open('./Cleaned_data/Final_v2/test_split.txt', 'r') as test_file:
        test_dataset = json.load(test_file)

    # Transform labels to Prompt format
    texts = train_dataset['text']
    texts.extend(validation_dataset['text'])
    texts.extend(test_dataset['text'])
    labels = train_dataset['label']
    labels.extend(validation_dataset['label'])
    labels.extend(test_dataset['label'])

    # Remove texts with ES labels and labels with ES labels
    texts_without_es = []
    for passage_idx in range(len(texts)):
        texts_without_es_passage = []
        for paragraph_idx in range(len(texts[passage_idx])):
            texts_without_es_passage.append([token for idx, token in enumerate(texts[passage_idx][paragraph_idx]) if labels[passage_idx][paragraph_idx][idx] not in ["B-ES", "I-ES"]])
        texts_without_es.append(texts_without_es_passage)

    labels_without_es = []
    for passage_idx in range(len(texts)):
        labels_without_es_passage = []
        for paragraph_idx in range(len(texts[passage_idx])):
            labels_without_es_passage.append([label for label in labels[passage_idx][paragraph_idx] if label not in ["B-ES", "I-ES"]])
        labels_without_es.append(labels_without_es_passage)
    
    # Verification
    for passage_idx in range(len(texts_without_es)):
        for paragraph_idx in range(len(texts_without_es[passage_idx])):
            if len(texts_without_es[passage_idx][paragraph_idx]) != len(labels_without_es[passage_idx][paragraph_idx]):
                print(passage_idx, paragraph_idx)

    texts = texts_without_es
    labels = labels_without_es
    
    for passage_idx in range(len(texts)):
        for paragraph_idx in range(len(texts[passage_idx])):
            for token_idx in range(len(texts[passage_idx][paragraph_idx])):
                token_label = labels[passage_idx][paragraph_idx][token_idx]
                next_token_label = "" if token_idx+1 >= len(texts[passage_idx][paragraph_idx]) else labels[passage_idx][paragraph_idx][token_idx+1]
                if token_label == "O" or (token_label.split('-')[0] == "I" and next_token_label.split('-')[0] == "I"):
                    labels[passage_idx][paragraph_idx][token_idx] = texts[passage_idx][paragraph_idx][token_idx]
                elif token_label.split('-')[0] == "B" and next_token_label.split('-')[0] != "I":
                    labels[passage_idx][paragraph_idx][token_idx] = "@@" + texts[passage_idx][paragraph_idx][token_idx] + f"({token_label.split('-')[1]})" + "@@"
                elif token_label.split('-')[0] == "B" and next_token_label.split('-')[0] == "I":
                    labels[passage_idx][paragraph_idx][token_idx] = "@@" + texts[passage_idx][paragraph_idx][token_idx]
                elif token_label.split('-')[0] == "I" and next_token_label.split('-')[0] != "I":
                    labels[passage_idx][paragraph_idx][token_idx] =  texts[passage_idx][paragraph_idx][token_idx] + f"({token_label.split('-')[1]})" + "@@"
    return texts, labels


async def dispatch_openai_requests(messages_list):
    async_responses = [
        openai.ChatCompletion.acreate(
            model="gpt-3.5-turbo",
            messages=messages
        )
        for messages in messages_list
    ]
    return await asyncio.gather(*async_responses)

async def dispatch_azure_openai_requests(messages_list):
    async_responses = [
        openai.ChatCompletion.acreate(
            engine="chaolab-gpt35-16k-useast2",
            messages=messages,
            temperature=0.7,
            max_tokens=1600,
            top_p=0.95,
            stop=None
        )
        for messages in messages_list
    ]
    return await asyncio.gather(*async_responses)

def prompt_gpt(texts, few_shot_n):
    result = []
    for passage_idx, passage in enumerate(tqdm(texts, desc="passage", position=0)):
        # Construct messages for all paragraphs within a passage and trigger openai APIs asynchronously
        messages_list = []
        for paragraph_idx, paragraph in enumerate(tqdm(passage, desc="paragraph", position=1, leave=False)):
            prompt_text = " ".join(paragraph) 
            messages = [
                {"role": "system", "content": generate_n_shot_prompt(few_shot_n)},
                {"role": "user", "content": f"Consider the following text:\n{prompt_text}"}
            ]           
            messages_list.append(messages)
        try:
            # openai_results = asyncio.run(dispatch_openai_requests(messages_list))
            openai_results = asyncio.run(dispatch_azure_openai_requests(messages_list))
        except:
            print("api error. Retry in 60 seconds.")
            time.sleep(60)
            try:
                # openai_results = asyncio.run(dispatch_openai_requests(messages_list))
                openai_results = asyncio.run(dispatch_azure_openai_requests(messages_list))
            except:
                openai_results = None

        # Read the result from the asynchronous calls
        try:
            result_passage = [openai_result['choices'][0]['message']['content'].split(" ") for openai_result in openai_results]
        except:
            print("Incorrect OpenAI Result:", openai_results)
            result_passage = [[]]
        result.append(result_passage)
    
        time.sleep(2)

    with open(f"./GPT/results/gpt35_fewshot_{few_shot_n}_ner.json", 'w+') as f:
        json.dump(result, f)
    return result


def process_labels(texts):
    processed_texts = []
    for passage_idx, passage in enumerate(texts):
        processed_passage = []
        for paragraph_idx, paragraph in enumerate(passage):
            processed_paragraph = []

            token_idx = 0
            while token_idx < len(paragraph):
                token = paragraph[token_idx]
                # We found an entity and finish marking it
                if token.startswith("@@"):
                    start_idx = token_idx
                    next_token_idx = token_idx + 1
                    while next_token_idx < len(paragraph) and not token.endswith("@@"):
                        token = paragraph[next_token_idx]
                        next_token_idx += 1
                    token_idx = next_token_idx - 1
                    entity_length = token_idx - start_idx + 1
                    token_type = re.search(r"\(([A-Za-z]*)\)@@$", paragraph[token_idx])
                    # GPT annotated token does not follow the correct format, treat the entire entity as "O"
                    if token_type == None:
                        processed_paragraph.extend(["O"] * entity_length)
                    # Annotated token has the correct format, add them to the processed_paragraph
                    else:
                        token_type = token_type.group()[1:-3]
                        processed_paragraph.extend([f"B-{token_type}"] + [f"I-{token_type}"] * (entity_length-1))
                # We did not find an entity
                else:
                    processed_paragraph.append("O")
                token_idx += 1
            processed_passage.append(processed_paragraph)
        processed_texts.append(processed_passage)
    return processed_texts


def compute_metrics(predictions, labels):
    # The metrics are calculated for each entity
    metrics = {}

    gpt_incorrect_length = 0
    for entity_type in ['CN', 'PN', 'PV', 'Condition']:
        true_positives = 0
        false_positives = 0
        false_negatives = 0

        for passage_idx, passage in enumerate(labels):
            if predictions[passage_idx] == [[]]:
                continue
            for paragraph_idx, paragraph in enumerate(passage):             
                prediction_paragraph = predictions[passage_idx][paragraph_idx]
                gold_paragraph = paragraph
                if len(prediction_paragraph) != len(gold_paragraph):
                    # print(f"GPT generated incorrect number of tokens during NER for passage_id {passage_idx} and paragraph_id {paragraph_idx}.")
                    gpt_incorrect_length += 1
                    continue
                for token_idx, token in enumerate(gold_paragraph):
                    predicted_token = prediction_paragraph[token_idx]
                    if token.endswith(entity_type) and predicted_token.endswith(entity_type):
                        true_positives += 1
                    elif token.endswith(entity_type) and not predicted_token.endswith(entity_type):
                        false_negatives += 1
                    elif not token.endswith(entity_type) and predicted_token.endswith(entity_type):
                        false_positives += 1

        precision = true_positives / (true_positives + false_positives + 0.01)
        recall = true_positives / (true_positives + false_negatives + 0.01)
        metrics[entity_type] = {"precision": precision, "recall": recall, "f1": (2 * precision * recall) / (precision + recall + 0.01), "tp": true_positives, "fp": false_positives, 'fn': false_negatives}
    print("Total GPT Incorrect Length:", gpt_incorrect_length / 4)
    return metrics

def generate_n_shot_prompt(n=1):
    base_prompt = "As a proficient linguist, your objective is to identify and label specific entities within a provided paragraph. These entities include chemical names (CN), property names (PN), property values (PV), and condition (Condition). Chemical names, polymer material names and their abstractions are entities. Polymer material names might contain multiple chemical names within it, label them as a single entity. Abstractions of property names are also considered entities. Property values contain both the number and the unit. To represent recognized named entities in the output text, enclose them within special symbols '@', followed by their respective types '(CN)', '(PN)', or '(PV)' before the ending '@'. The remaining text should remain unchanged. Below are some examples: "
    for i in range(n):
        base_prompt += f"\n{getattr(prompt_examples, f'EXAMPLE{i+1}')}\n{getattr(prompt_examples, f'ANSWER{i+1}')}"
    return base_prompt

if __name__ == '__main__':
    texts, labels = data_preprocessor()

    # Prediction with GPT
    # prompt_gpt(texts, 9)

    # Run Evaluation metrics
    with open('./GPT/results/gpt35_fewshot_9_ner.json', 'r') as f:
        predictions = json.load(f)
    # print(predictions[0][1])
    prediction_labels = process_labels(predictions)
    gold_labels = process_labels(labels)
    metrics = compute_metrics(prediction_labels, gold_labels)
    print(metrics)

    # Generate Prompt Examples
    # print(" ".join(labels[79][2]))
    # print(" ".join(texts[79][2]))
