import json
import pathlib
import time
import argparse
from base import openai, SLEEP_SECONDS


def ask_model_to_execute_the_code(
        dataset_path,
        code_representation_base_path,
        code_execution_base_path,
        engine,
        temperature,
):
    with open(dataset_path, 'r') as aggregated_boxes_file:
        aggregated_boxes = aggregated_boxes_file.readlines()

    for json_str in aggregated_boxes:
        data = json.loads(json_str)
        sentence_hash = data['sentence_hash']

        specific_code_representation_path = pathlib.Path(f"{code_representation_base_path}/{sentence_hash}.py")
        specific_code_execution_path = pathlib.Path(f"{code_execution_base_path}/{sentence_hash}.txt")

        if specific_code_representation_path.is_file() and not specific_code_execution_path.is_file():
            with open(specific_code_representation_path, 'r') as code_file:
                code = code_file.read()
            code_prompt = "Run the following code and print the results:\n\n" + code
            try:
                response = openai.ChatCompletion.create(
                    model=engine,
                    messages=[
                        {"role": "user", "content": code_prompt},
                    ],
                    temperature=temperature,
                )
                output = response['choices'][0]['message']['content']

                with open(specific_code_execution_path, 'w') as d:
                    d.write(output)
                print(sentence_hash, "finished")
            except openai.error.OpenAIError as e:
                print(e)
                print("sleeping")
                time.sleep(SLEEP_SECONDS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process and execute code using OpenAI.')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the input aggregated data.')
    parser.add_argument('--code_representation_base_path', type=str, required=True,
                        help='Base path for code representations.')
    parser.add_argument('--code_execution_base_path', type=str, required=True,
                        help='Base path for code execution outputs.')
    parser.add_argument('--engine', type=str, default="gpt-3.5-turbo", help='OpenAI engine to use.')
    parser.add_argument('--temperature', type=float, default=0, help='Temperature setting for OpenAI model.')

    args = parser.parse_args()

    ask_model_to_execute_the_code(
        args.dataset_path,
        args.code_representation_base_path,
        args.code_execution_base_path,
        args.engine,
        args.temperature,
    )
