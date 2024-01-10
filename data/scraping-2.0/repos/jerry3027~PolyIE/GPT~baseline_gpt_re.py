import re
import json
import time
import openai
import asyncio
from tqdm import tqdm
from GPT import prompt_examples

# OpenAI API key
# openai.api_key = "***"

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
    relation = train_dataset['relation']
    relation.extend(validation_dataset['relation'])
    relation.extend(test_dataset['relation'])

    # Convert relation labels to machine readable format
    processed_relation = []
    for passage_idx in range(len(relation)):
        processed_relation_passage = []
        for paragraph_idx in range(len(relation[passage_idx])):
            processed_relation_paragraph = []
            for relation_idx in range(len(relation[passage_idx][paragraph_idx])):
                current_relation = []
                for entity_idx in relation[passage_idx][paragraph_idx][relation_idx]:
                    entity_start = entity_idx[0]
                    entity_end = entity_idx[1]
                    current_relation.append(" ".join(texts[passage_idx][paragraph_idx][entity_start:entity_end]))
                processed_relation_paragraph.append(current_relation)
            processed_relation_passage.append(processed_relation_paragraph)
        processed_relation.append(processed_relation_passage)

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
    return texts, labels, processed_relation


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
            frequency_penalty=0,
            presence_penalty=0,
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

    with open(f"./GPT/results/gpt35_fewshot_{few_shot_n}_re.json", 'w+') as f:
        json.dump(result, f)
    return result


def process_prediction(labels):
    processed_labels = []
    for passage_idx, passage in enumerate(labels):
        processed_passage = []
        for paragraph_idx, paragraph in enumerate(passage):
            relations = [x.group() for x in re.finditer(r"\((.*,.*, .*)\)", " ".join(paragraph))]
            processed_relations = []
            for relation in relations:
                processed_relations.append(tuple(i for i in relation.strip('()').split(", ")))
            processed_passage.append(processed_relations)
        processed_labels.append(processed_passage)
    return processed_labels


def compute_metrics(predictions, labels):
    # The metrics are calculated for each entity
    metrics = {}
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for passage_idx, passage in enumerate(labels):
        if predictions[passage_idx] == [[]]:
            continue
        for paragraph_idx, paragraph in enumerate(passage):
            prediction_paragraph = predictions[passage_idx][paragraph_idx]
            gold_paragraph = paragraph

            curr_tp = 0

            for i in range(len(gold_paragraph)):
                for j in range(len(prediction_paragraph)):
                    gold_rel = gold_paragraph[i]
                    predicted_rel = prediction_paragraph[j]
                    matched = 0
                    if len(predicted_rel) < len(gold_rel):
                        continue
                    for k in range(len(gold_rel)):
                        if gold_rel[k] in predicted_rel[k]:
                            matched += 1
                    if matched == len(gold_rel):
                        curr_tp += 1
                        break
            
            curr_fp = len(prediction_paragraph) - curr_tp
            curr_fn = len(gold_paragraph) - curr_tp
            
            true_positives += curr_tp
            false_positives += curr_fp
            false_negatives += curr_fn

    precision = true_positives / (true_positives + false_positives + 0.01)
    recall = true_positives / (true_positives + false_negatives + 0.01)
    metrics = {"precision": precision, "recall": recall, "f1": (2 * precision * recall) / (precision + recall + 0.01), "tp": true_positives, "fp": false_positives, 'fn': false_negatives}
    return metrics

def generate_n_shot_prompt(n=1):
    base_prompt = f'As a skilled linguist, your mission is to analyze a provided paragraph that contains four distinct types of entities: Chemical Names (CN), Property Names (PN), Property Values (PV), and Conditions (Condition). Each of these entities is enclosed within "@" symbols, with their entity type specified in brackets before the closing "@". Your objective is to identify and extract relationships among these entities, and then present them in one of two possible formats: (Chemical Names, Property Names, Property Values, Condition) or (Chemical Names, Property Names, Property Values). Please only establish relationships using the provided entities, and only provide a list of the extracted relations. Below are some examples: '
    for i in range(n):
        base_prompt += f"\n{getattr(prompt_examples, f'RE_EXAMPLE_{i+1}')}\n{getattr(prompt_examples, f'RE_ANSWER_{i+1}')}"
    return base_prompt

if __name__ == '__main__':
    texts, labels, relations = data_preprocessor()

    # # Prediction with GPT
    # prompt_gpt(texts,9)

    # Run Evaluation metrics
    with open('./GPT/results/gpt35_fewshot_9_re.json', 'r') as f:
        predictions = json.load(f)
    prediction_labels = process_prediction(predictions)
    gold_labels = relations
    metrics = compute_metrics(prediction_labels, gold_labels)
    print(metrics)

    # Generate Prompt Examples
    # print(" ".join(labels[0][2]))
    # print([tuple(i) for i in relations[0][2]])

    # print(" ".join(labels[1][4]))
    # print([tuple(i) for i in relations[1][4]])

    # print(" ".join(labels[11][4]))
    # print([tuple(i) for i in relations[11][4]])

    # print(" ".join(labels[21][2]))
    # print([tuple(i) for i in relations[21][2]])

    # print(" ".join(labels[51][2]))
    # print([tuple(i) for i in relations[51][2]])

    # print(" ".join(labels[63][2]))
    # print([tuple(i) for i in relations[63][2]])

    # print(" ".join(labels[72][3]))
    # print([tuple(i) for i in relations[72][3]])

    # print(" ".join(labels[82][4]))
    # print([tuple(i) for i in relations[82][4]])

    # print(" ".join(labels[41][3]))
    # print([tuple(i) for i in relations[41][3]])

    # print(" ".join(labels[2][4]))
    # print([tuple(i) for i in relations[2][4]])
