import openai
from dotenv import load_dotenv
import os
import json
import time

load_dotenv()

openai.api_key = os.environ.get("OPENAI_API_KEY")

# for making RM dataset
output = []
DATASET_PATH = "./data/aihub/kakao_cleaned.jsonl"
OUTPUT_PATH = "./data/aihub/kakao_light_rm.jsonl"
START_IDX = 1001
END_IDX = 6000

# for OpenAI API Request
start_sequence = "\nAI:"
restart_sequence = "\nHuman: "
ada_prompt = "The following is a conversation with an AI friend in messenger application. He is very friendly and talks like real friends. He must talk in Korean. He speaks casually.\n"
chatgpt_prompt = "당신은 제 친구이며, 메신저로 대화하는 상황을 연기하세요. 당신은 반드시 한국어로 답변해야 합니다. 존댓말을 사용하지 말고, 반말로 대답해주세요."


def make_prompt(prev_q, prev_r, cur_q):
    return f"{ada_prompt}\nHuman: {prev_q}\nAI: {prev_r}\nHuman: {cur_q}\nAI: "


if __name__ == "__main__":
    with open(DATASET_PATH, "r") as r:
        dataset = json.load(r)

    with open(OUTPUT_PATH, "r") as r:
        output = json.load(r)

    # 리소스 이슈로 인해 END_IDX 개의 대화에만 진행
    for idx in range(START_IDX, END_IDX):
        try:
            prev_query = dataset[idx - 2]['query']
            prev_response = dataset[idx - 1]['query']
            cur_query = dataset[idx]['query']
            cur_response = dataset[idx]['response']

            prompt = make_prompt(prev_query, prev_response, cur_query)
            messages = [
                {"role": "system", "content": chatgpt_prompt},
                {"role": "user", "content": prev_query},
                {"role": "assistant", "content": prev_response},
                {"role": "user", "content": cur_query}
            ]

            # OpenAI api request(text-ada-001)
            ada_response = openai.Completion.create(
                model="text-ada-001",
                prompt=prompt,
                temperature=0.9,
                max_tokens=64,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0.6,
                stop=[" Human:", " AI:"]
            )['choices'][0]['text']

            # OpenAI api request(text-curie-001)
            # curie_response = openai.Completion.create(
            #     model="text-curie-001",
            #     prompt=prompt,
            #     temperature=0.9,
            #     max_tokens=64,
            #     top_p=1,
            #     frequency_penalty=0,
            #     presence_penalty=0.6,
            #     stop=[" Human:", " AI:"]
            # )['choices'][0]['text']

            chatgpt_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=64
            )['choices'][0]['message']['content']

            candidates = ['', '', '']
            ranking = [0, 1, 2]
            answer_idx, chatgpt_idx, ada_idx = (
                idx % 3, (idx + 1) % 3, (idx + 2) % 3)

            candidates[answer_idx] = cur_response
            candidates[chatgpt_idx] = chatgpt_response
            candidates[ada_idx] = ada_response
            ranking[answer_idx] = 0
            ranking[chatgpt_idx] = 1
            ranking[ada_idx] = 2

            rm_data = {
                "prompt": cur_query,
                f"completion_{answer_idx}": cur_response,
                f"completion_{chatgpt_idx}": chatgpt_response,
                f"completion_{ada_idx}": ada_response,
                "ranking": ranking
            }

            print(f"{idx} done")
            output.append(rm_data)

            # Chatgpt API가 불안정해서 100번째 대화마다 저장
            if idx % 100 == 0:
                with open(OUTPUT_PATH, "w", encoding='utf-8') as wr:
                    json.dump(output, wr, indent=4, ensure_ascii=False)

        except Exception as e:
            print(e)
            time.sleep(10)
            continue
