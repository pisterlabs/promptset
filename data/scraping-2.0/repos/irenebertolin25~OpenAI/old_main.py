import os
import json
import jsonlines
import time
import csv
from dotenv import load_dotenv
from openai import OpenAI
from collections import defaultdict
from pydantic import BaseModel
from io import StringIO

load_dotenv()

# Set your API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def valid_data_jsonl(data_path, isTrainingData):
    print("\033[0m" + "Starting valid_data_jsonl...")
    try:
        # Load the dataset
        with open(data_path, 'r', encoding='utf-8') as f:
            dataset = [json.loads(line) for line in f]

        # Initial dataset stats
        print("Num examples:", len(dataset))
        print("First example:")
        for message in dataset[0]["messages"]:
            print(message)

        # Format error checks
        format_errors = defaultdict(int)

        for ex in dataset:
            if not isinstance(ex, dict):
                format_errors["data_type"] += 1
                continue

            messages = ex.get("messages", None)
            if not messages:
                format_errors["missing_messages_list"] += 1
                continue

            for message in messages:
                if "role" not in message or "content" not in message:
                    format_errors["message_missing_key"] += 1

                if any(k not in ("role", "content", "name", "function_call") for k in message):
                    format_errors["message_unrecognized_key"] += 1

                if message.get("role", None) not in ("system", "user", "assistant", "function"):
                    format_errors["unrecognized_role"] += 1

                content = message.get("content", None)
                function_call = message.get("function_call", None)

                if (not content and not function_call) or not isinstance(content, str):
                    format_errors["missing_content"] += 1

            if isTrainingData:
                if not any(message.get("role", None) == "assistant" for message in messages):
                    format_errors["example_missing_assistant_message"] += 1

        if format_errors:
            print("Found errors:")
            for k, v in format_errors.items():
                print(f"{k}: {v}")
            return False

    except FileNotFoundError:
        print("\033[91m\u2718 " + f"The file '{data_path}' was not found.")
        return False

    except Exception as e:
        print("\033[91m\u2718 " + f"An error occurred: {e}")
        return False

    print("\033[92m\u2714 " + "No errors found")
    return True

def add_prompt_to_jsonl_file(input_file, output_file, prompt):
    print("\033[0m" + "Starting add_prompt_to_jsonl_file...")
    try:
        with jsonlines.open(input_file, 'r') as reader, jsonlines.open(output_file, 'w') as writer:
            for line in reader:
                messages = line.get('messages', [])
                if messages:
                    messages.insert(0, {'role': 'system', 'content': prompt})
                line['messages'] = messages
                writer.write(line)
    
    except Exception as e:
        print("\033[91m\u2718 " + f"An error occurred: {e}")
        return None

    print("\033[92m\u2714 " + f"Jsonl dataset with prompt saved in {output_file}")
    return output_file

def upload_file(file_path):
    print("\033[0m" + "Starting upload_file...")
    try:
        file_uploaded = client.files.create(file=open(file_path, "rb"), purpose='fine-tune')

        while True:
            print("Waiting for file to process...")
            file_handle = client.files.retrieve(file_uploaded.id)
            if file_handle and file_handle.status == "processed":
                print("\033[92m\u2714 " + "File processed")
                break
            time.sleep(120)

    except Exception as e:
        print("\033[91m\u2718 " + f"An error occurred: {e}")
        return None

    print("\033[92m\u2714 " + "File uploaded")
    return file_uploaded

def generate_fine_tuning_results(path, file_name, content):
    print("\033[0m" + "Starting generate_fine_tuning_results...")
    try:
        directory = os.path.join(path, os.path.dirname(file_name))
        if not os.path.exists(directory):
            os.makedirs(directory)

        existing_files = [item_file for item_file in os.listdir(path) if item_file.startswith(file_name) and item_file.endswith('.json')]
        next_number = 0 if not existing_files else len(existing_files) + 1
        file_name = f'{file_name}{next_number}.json'

        with open(os.path.join(path, file_name), 'w') as result_file:
            json.dump(content, result_file, indent=4)

    except Exception as e:
        print("\033[91m\u2718 " + f"An error occurred: {e}")
        return None

    print("\033[92m\u2714 " + f"Results saved in {path}{file_name}")
    return result_file

def create_csv_file(path, file_name, content):
    print("\033[0m" + "Starting create_csv_file...")
    try:
        directory = os.path.join(path, os.path.dirname(file_name))
        if not os.path.exists(directory):
            os.makedirs(directory)

        existing_files = [item_file for item_file in os.listdir(path) if item_file.startswith(file_name) and item_file.endswith('.csv')]
        next_number = 0 if not existing_files else len(existing_files) + 1
        file_name = f'{file_name}{next_number}.csv'

        lines = content.split('\n')
        data = [line.split(',') for line in lines]

        with open(os.path.join(path, file_name), 'w', newline='') as result_file:
            csv_writer = csv.writer(result_file)
            csv_writer.writerows(data)
    
    except Exception as e:
        print("\033[91m\u2718 " + f"An error occurred: {e}")
        return None

    print("\033[92m\u2714 " + f"Results saved in {path}{file_name}")
    return result_file

def create_csv_file_from_array(path, file_name, content):
    print("\033[0m" + "Starting create_csv_file_from_array...")
    try:
        directory = os.path.join(path, os.path.dirname(file_name))
        if not os.path.exists(directory):
            os.makedirs(directory)

        existing_files = [item_file for item_file in os.listdir(path) if item_file.startswith(file_name) and item_file.endswith('.csv')]
        next_number = 0 if not existing_files else len(existing_files) + 1
        file_name = f'{file_name}{next_number}.csv'

        fieldnames = ['User review', 'Emotion for chat completion with temperature 0.2', 'Emotion for chat completion with temperature 0.8']

        with open(os.path.join(path, file_name), 'w', newline='') as result_file:
            writer = csv.writer(result_file)
            writer.writerow(fieldnames)
            writer.writerows(content)
    
    except Exception as e:
        print("\033[91m\u2718 " + f"An error occurred: {e}")
        return None

    print("\033[92m\u2714 " + f"Results saved in {path}{file_name}")
    return result_file

def create_fine_tuning_job(file_uploaded, model):
    print("\033[0m" + "Starting create_fine_tuning_job...")
    try:
        fine_tuning_job = client.fine_tuning.jobs.create(training_file=file_uploaded.id, model=model)

        status = fine_tuning_job.status
        if status not in ["succeeded", "failed"]:
            print("Waiting for fine-tuning to complete...")
            while status not in ["succeeded", "failed"]:
                time.sleep(60)
                fine_tuning_job = client.fine_tuning.jobs.retrieve(fine_tuning_job.id)
                status = fine_tuning_job.status
                print("Status: ", status)
        
        print("\033[92m\u2714 " + f"Fine-tune job {fine_tuning_job.id} finished with status: {status}")

    except Exception as e:
        print("\033[91m\u2718 " + f"An error occurred: {e}")
        return None

    fine_tuning_job_serializable = fine_tuning_job.model_dump()
    generate_fine_tuning_results("results/", "fine_tuning_job_result_", fine_tuning_job_serializable)

    if generate_fine_tuning_results is None:
            print("\033[91m\u2718 " + "Error in generate_fine_tuning_results")
            return None

    return fine_tuning_job

def chat_completion(model, messages, temperature):
    print("\033[0m" + "Starting chat_completion...")
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature
        )

    except Exception as e:
        print("\033[91m\u2718 " + f"An error occurred: {e}")
        return None

    print("\033[92m\u2714 " + f'Completion with temperature {temperature}: ', completion.choices[0].message.content)
    return completion

def convert_to_serializable(item):
    try:
        if isinstance(item, set):
            return list(item)
        else:
            return item
    
    except Exception as e:
        print("\033[91m\u2718 " + f"An error occurred: {e}")
        return None

def generate_chat_completion_results(model, dataset_path):
    print("\033[0m" + "Starting generate_chat_completion_results...")
    try:
        generated_completion = []
        with open(dataset_path, "r") as test_data:
            for line in test_data:
                data = json.loads(line)
                data_message = data.get("messages")
                user_review = data_message[1].get("content")
                print("\033[0m" + 'User review: ', user_review)

                completion_low_temperature = chat_completion(model, data_message, 0.2)
                if completion_low_temperature is None:
                    print("\033[91m\u2718 " + "Error in chat completion with temperature 0.2")
                    return None
                completion_high_temperature = chat_completion(model, data_message, 0.8)
                if completion_high_temperature is None:
                    print("\033[91m\u2718 " + "Error in chat completion with temperature 0.8")
                    return None

                result = [user_review, completion_low_temperature.choices[0].message.content, completion_high_temperature.choices[0].message.content]
                generated_completion.append(result)
        
        generated_completion = [convert_to_serializable(item) for item in generated_completion]
        if generated_completion is None:
            print("\033[91m\u2718 " + "Error in convert_to_serializable")
            return None

        chat_completion_result = create_csv_file_from_array("results/", "chat_completion_result_", generated_completion)
        if chat_completion_result is None:
            print("\033[91m\u2718 " + "Error in create_csv_file")
            return None
        
        return chat_completion_result

    except Exception as e:
        print("\033[91m\u2718 " + f"An error occurred: {e}")
        return None

def refactor_id_jsonl(jsonl_file):
    print("\033[0m" + "Starting refactor_id_jsonl...")
    try:
        id_count = {}
        updated_lines = []

        with open(jsonl_file, 'r') as reader:
            lines = reader.readlines()

        for line in lines:
            data = json.loads(line)
            current_id = data.get('id')

            if current_id in id_count:
                id_count[current_id] += 1
                data['id'] = f"{current_id}_{id_count[current_id]}"
            else:
                id_count[current_id] = 0

            updated_lines.append(json.dumps(data))

        with open(jsonl_file, 'w') as writer:
            for updated_line in updated_lines:
                writer.write(f"{updated_line}\n")

    except Exception as e:
        print("\033[91m\u2718 " + f"An error occurred: {e}")
        return None

    print("\033[92m\u2714 " + f'IDs processed in {jsonl_file}')

def add_emotion_and_id_to_csv(jsonl_file, csv_file):
    print("\033[0m" + "Starting add_emotion_and_id_to_csv...")
    try:
        emotions = {}
        ids = []

        with open(csv_file.name, 'r+', newline='', encoding='utf-8') as file:
            csv_data = StringIO(file.read())
            csv_data.seek(0)

            csv_reader = csv.reader(csv_data)
            csv_list = list(csv_reader)

            header = csv_list[0]
            header.insert(0, 'ID')
            header.append('Emotion label')

            csv_data.seek(0)

            with jsonlines.open(jsonl_file, 'r') as reader:
                for line in reader:
                    ids.append(line["id"])
                    emotions[line["id"]] = line["emotion"]

            for i, row in enumerate(csv_list[1:], start=1):
                row.insert(0, ids[i - 1])
                emotion = emotions.get(ids[i - 1], '')
                row.append(emotion)

            csv_data.seek(0)

            writer = csv.writer(csv_data)
            writer.writerows(csv_list)

            file.seek(0)
            file.write(csv_data.getvalue())
            file.truncate()

    except Exception as e:
        print("\033[91m\u2718 " + f"An error occurred: {e}")
        return None

    print("\033[92m\u2714 " + "IDs and emotions added")
    return csv_file

def main():
    print("Starting main...")
    # NOTE: You can change this to any dataset you want to fine-tune on
    training_dataset = "./data/training_dataset.jsonl"
    # NOTE: You can change this to any dataset you want to test
    test_dataset = "./data/test_dataset.jsonl"
    
    if not valid_data_jsonl(training_dataset, True) or not valid_data_jsonl(test_dataset, False):
        return

    # Add prompt to training file
    system_prompt = "Can you tell me what emotion is expressing the message above? It can be one of the following: happiness, sadness, anger, fear, surprise, disgust or not-relevant (if it doesn't express any relevant emotion)."
    training_dataset_with_prompt = add_prompt_to_jsonl_file(training_dataset, "data/training_dataset_with_prompt.jsonl", system_prompt)
    if training_dataset_with_prompt is None:
        return

    # Upload training file
    file_uploaded = upload_file(training_dataset_with_prompt)
    if file_uploaded is None:
        return

    # Create fine-tuning job
    model = "gpt-3.5-turbo"
    fine_tuned_job = create_fine_tuning_job(file_uploaded, model)
    if fine_tuned_job is None:
        return

    # Get the metrics
    result_files_id = fine_tuned_job.result_files[0]
    content = client.files.retrieve_content(result_files_id)
    csv_file = create_csv_file("results/", "training_metrics_", content)
    if csv_file is None:
        return

    # Add prompt to test file
    test_dataset_with_prompt = add_prompt_to_jsonl_file(test_dataset, "data/test_dataset_with_prompt.jsonl", system_prompt)
    if test_dataset_with_prompt is None:
        return

    # Create chat completion and generate result file
    fine_tuned_model = fine_tuned_job.fine_tuned_model
    csv_results_file = generate_chat_completion_results(fine_tuned_model, test_dataset_with_prompt)
    if csv_results_file is None:
        return

    # Add ID & emotion to csv file
    emotion_labels_path = "./data/emotion_labels_test_dataset.jsonl"
    refactor_id_jsonl(emotion_labels_path)
    csv_results_file = add_emotion_and_id_to_csv(emotion_labels_path, csv_results_file)
    if csv_results_file is None:
        return
    
    print("\033[92m\u2714 " + "The program has finished successfully")

if __name__ == '__main__':
    main()