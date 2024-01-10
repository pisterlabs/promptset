import numpy as np
from tqdm.auto import tqdm
import json
import requests
from json import JSONDecodeError
import traceback
import sys
import os

dataset_path = sys.argv[1]

with open(dataset_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)


def get_query(question, context, model_output):
    query = (
        f"Here is a question:\n{question}\n\n"
        f"Here is a piece of text that contains some information related to the above question:\n{context}\n\n"
        f"Here is a variant of answer that you need to evaluate:\n{model_output}\n\n"
        "You need to decide if it is acceptable according to the information provided in the related piece of text or not. "
        "An acceptable answer should contain only true information and should be complete (according to the related piece of text). "
        "Don't be too picky; the main purpose is to verify that the answer does not contain made-up information that is not in the piece of text provided, or information that contradicts that in the piece of text provided."
        "Write just a word yes or no, do not write anything else."
    )
    return query

def get_model_output(question, prompt_techniques):

    num_attempts = 0
    request_status = False

    while not request_status:
        try:
            x = requests.post('http://127.0.0.1:8000/', json={'question': question, 'techniques': prompt_techniques})
            model_output = json.loads(x.text)['answer']['answer']
            request_status = True
        except JSONDecodeError:
            num_attempts += 1
            print("new attempt")
            if num_attempts > 1:
                raise RuntimeError

    return model_output

def print_to_file_and_terminal(*args, file):
    print(*args)
    print(*args, file=file)
    file.flush()

with open('logs/quality_metrics_mistral.txt', 'a') as log_file:
    # for technique_list in [['cot', 'few-shot', 'reit', 'right', 'rawinst', 'pos'], ['cot', 'few-shot'], ['cot'], ['few-shot'], []]:
    for technique_list in [[]]:
        try:
            print_to_file_and_terminal('Start evaluating...', file=log_file)
            print_to_file_and_terminal(technique_list, file=log_file)
            N = len(data)
            assert N != 0
            # N = 10
            num_acceptable_answers = 0
            num_failed_attempts = 0

            for i in tqdm(range(N)):
                # print(i)
                response_status = False
                num_attempts = 0

                try:
                    model_output = get_model_output(data[i]['question'], technique_list)
                    with open('logs/qa_logs_mistral.txt', 'a') as qa_logs_file:
                        print(
                            json.dumps({
                                'question': data[i]['question'],
                                'prompt_techniques': technique_list,
                                'model_output': model_output
                            }), 
                            file=qa_logs_file
                            )
                        qa_logs_file.flush()
                except KeyboardInterrupt:
                    print('Interrupted')
                    sys.exit(0)
                except:
                    num_failed_attempts += 1
                    continue

                while not response_status:

                    num_attempts += 1

                    response = client.chat.completions.create(
                        messages=[
                        {"role": "system", "content": "You're the chief editor of a news agency in Switzerland. You're fluent in French. You have a strong critical mind and always respond accurately and concisely. You can't be wrong in your answers, as it would ruin your reputation."},
                        {"role": "user", "content": get_query(question = data[i]['question'],
                                                            context = data[i]['transcript'],
                                                            model_output = model_output)}
                    ],
                        model="gpt-4-1106-preview",
                    )

                    response = response.choices[0].message.content.lower()

                    if "yes" in response:
                        num_acceptable_answers += 1
                        response_status = True

                    elif "no" in response:
                        response_status = True

                    else:
                        if num_attempts > 5:
                            num_failed_attempts += 1
                            response_status = True

                if (i + 1) % 20 == 0:
                    print_to_file_and_terminal(f"intermediate result (first {i + 1} samples):", file=log_file)
                    print_to_file_and_terminal(
                        f"Correct answer rate: {num_acceptable_answers / (i + 1 - num_failed_attempts):.2f}", file=log_file)
                    print_to_file_and_terminal(f"Failed attemps rate: {num_failed_attempts / (i + 1):.2f}, {N=}", file=log_file)

            print_to_file_and_terminal("Final for prompt techniques:", technique_list, file=log_file)
            print_to_file_and_terminal(
                f"Correct answer rate: {num_acceptable_answers / (N - num_failed_attempts):.2f}", file=log_file)
            print_to_file_and_terminal(f"Failed attemps rate: {num_failed_attempts / N:.2f}, {N=}", file=log_file)
            
        except Exception as e:
            print_to_file_and_terminal('error for', technique_list, file=log_file)
            traceback.print_exc(file=log_file)
            continue