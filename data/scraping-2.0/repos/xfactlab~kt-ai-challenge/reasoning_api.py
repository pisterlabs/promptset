import openai
import yaml
from src import utils
import time
from pprint import pprint
import os
import argparse
from tqdm import tqdm


def get_message(prompt):
    """
    Message for chat api
    """
    message = {
        "role" : "user",
        "content" : prompt
        }
    return [message]


def generate_answer(message, model_name, max_token, temperature):
    """
    Generate response
    """
    try:
        completion = openai.ChatCompletion.create(
            model=model_name,
            messages=message,
            temperature=temperature,
            max_tokens=max_token,
            n=1)
    except:
        print("retrying due to an error......")
        time.sleep(20)
        return generate_answer(message, model_name, max_token, temperature)

    return completion


def get_reason_mgs(question, answer):
    """
    Message for chat api
    주어진 질문과 답에 대해, 풀이를 수식으로 300자이내로 서술하시오.
    질문: 지름이 12cm, 높이가 14cm인 원기둥의 전개도를 그렸을 때 전개도의 옆면의 둘레는 몇 cm인지 답을 구하시오. (원주율: 3) 답: 100
    """

    msg = [
        {
            "role": "system",
            "content": "주어진 질문과 정답에 대해, 풀이를 수식으로 300자이내로 서술하시오."
        },
        {
            "role": "user",
            "content": f"질문: {question} 정답:{answer}"
        }
    ]

    return msg


def main(args):

    api_key_file = args.api_key_file

    model_name = args.model_name
    max_token = args.max_token
    temperature = args.temperature

    input_file = args.input_file
    output_file = args.output_path

    # Adding openai key
    key_dict = utils.get_keys(api_key_file)
    openai.api_key = key_dict['openai']

    # Create output_file & initializing file writer
    print(f'output_file : {output_file}')
    jsonl_writer = utils.JSONLWriter(output_file)

    # Reading input_file
    data = utils.read_jsonlines(input_file)

    for instance in tqdm(data):
        query = instance['question']
        ans = instance['answer']

        query_msg = get_reason_mgs(query, ans)

        query_completion = generate_answer(query_msg, model_name, max_token, temperature)
        q_response = query_completion["choices"][0]["message"]["content"]

        instance['reason'] = q_response
        jsonl_writer.write_json_line(instance)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Eval script for GSM8K")

    # API information
    parser.add_argument("--api_key_file", type=str, default='/userhomes/philhoon/self-debate/api.yaml',
                        help="Path to the API key file")

    parser.add_argument("--model_name", type=str, default="gpt-3.5-turbo-0301", help="The model name")
    parser.add_argument("--max_token", type=int, default=2048, help="max_token_length")
    parser.add_argument("--temperature", type=float, default=0, help="model temperature")

    # Input information
    parser.add_argument("--input_file", type=str, default='/projects/self-debate/data/gsm8k/test.jsonl', help="Path to the input file")

    # Output location
    parser.add_argument("--output_path", type=str, default='/userhomes/philhoon/self-debate/result', help="Path to the output directory")

    args = parser.parse_args()
    main(args)

# python openai_mwp_add_reasoning_api.py \
# --api_key_file /userhomes/philhoon/kt-ai-challenge/api-openai.yaml \
# --model_name gpt-3.5-turbo-0613 \
# --max_token 2048 \
# --temperature 0 \
# --input_file /userhomes/philhoon/kt-ai-challenge/data/public_mwp_data_v2_preprocess.jsonl \
# --output_path /userhomes/philhoon/kt-ai-challenge/data/public_mwp_data_v2_reasoning.jsonl
