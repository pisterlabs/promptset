from concurrent.futures import ThreadPoolExecutor
import json
import os
from models.openai_chat_model import OpenAIChatModel
from datasets import load_dataset
from tqdm import tqdm


def add_question_to_row(model, row):
    # Already computed
    if "questions" in row["context"]:
        return row

    def generate_question(sentence):
        return model.generate_question(sentence)

    all_questions = []

    # Create a single ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=100) as executor:
        # Store futures for each sentence in a dictionary to maintain order
        futures_dict = {}
        for paragraph_index, paragraph in enumerate(row["context"]["content"]):
            for sentence_index, sentence in enumerate(paragraph):
                future = executor.submit(generate_question, sentence)
                futures_dict[(paragraph_index, sentence_index)] = future

        # Organize the results into the structure of paragraphs and sentences
        for paragraph_index, paragraph in enumerate(row["context"]["content"]):
            paragraph_questions = []
            for sentence_index, _ in enumerate(paragraph):
                future = futures_dict[(paragraph_index, sentence_index)]
                paragraph_questions.append(future.result())
            all_questions.append(paragraph_questions)

    row["context"]["questions"] = all_questions

    return row


def add_paraphrased_question_to_row(model, row):
    # Already computed
    if "paraphrased_questions" in row["context"]:
        return row

    def generate_paraphrase(sentence):
        return model.generate_paraphrase(sentence)

    question_lut = {}
    for title, questions in zip(row["context"]["title"], row["context"]["questions"]):
        sent_counter = 0

        for question in questions:
            question_lut[(title, sent_counter)] = question
            sent_counter += 1

    paraphrased_questions = []
    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = []
        for title, sent_id in zip(
            row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"]
        ):
            key = (title, sent_id)
            if key not in question_lut:
                # In case of a bad pointer, skip it
                continue
            question = question_lut[key]
            future = executor.submit(generate_paraphrase, question)
            futures.append(future)

        for future in futures:
            paraphrased_questions.append(future.result())

    row["context"]["paraphrased_questions"] = paraphrased_questions

    return row


def convert_to_question_for_split(dataset, model, split, debug):
    split_path = f"data/2wikimultihopqa_with_q_gpt35/{split}.jsonl"

    TRAIN_LIMIT = 15000

    processed_ids = set()
    if os.path.exists(split_path):
        with open(split_path) as f:
            for line in f:
                row = json.loads(line)
                processed_ids.add(row["_id"])

    with open(split_path, "a") as f:
        for current_row, row in enumerate(
            tqdm(dataset[split], total=min(TRAIN_LIMIT, len(dataset[split])))
        ):
            if debug and current_row >= 1:
                break

            if row["_id"] in processed_ids:
                continue

            # Limit for trainset for now
            if current_row >= TRAIN_LIMIT:
                break

            add_question_to_row(model, row)

            add_paraphrased_question_to_row(model, row)

            f.write(json.dumps(row) + "\n")
            f.flush()


def convert_to_question_dataset(model, debug=False):
    dataset = load_dataset("somebody-had-to-do-it/2WikiMultihopQA")

    convert_to_question_for_split(dataset, model, "train", debug)
    convert_to_question_for_split(dataset, model, "dev", debug)


if __name__ == "__main__":
    config = {
        "architecture": {
            "question_generator_model": {
                "name": "gpt-3.5-turbo",
                # "name": "t5",
                # "size": "base",
                # "device": "cuda:0",
            }
        }
    }
    # model = T5ModelForQuestionGeneration(config)
    model = OpenAIChatModel(config)
    convert_to_question_dataset(model, debug=False)
