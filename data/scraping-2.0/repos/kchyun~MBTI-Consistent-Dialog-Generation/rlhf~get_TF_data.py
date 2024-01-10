import openai
from dotenv import load_dotenv
import os
import json
import time

load_dotenv()

# 프로젝트 루트 폴더에 .env 파일 생성 및 OPENAI_API_KEY={본인 api key} 항목 추가
openai.api_key = os.environ.get("OPENAI_API_KEY")

# for making RM dataset
output = []

MBTI_TF = 'T'
DATASET_PATH = "./data/aihub/instagram_cleaned.jsonl"
OUTPUT_PATH = f"./data/mbti/instagram_chatgpt_{MBTI_TF}.jsonl"
START_IDX = 3
END_IDX = 10003


if __name__ == "__main__":
    chatgpt_prompt = f"당신은 제 친한 친구이며, 저와 메신저로 대화하는 상황입니다. 당신의 MBTI 세 번째 유형은 {MBTI_TF}이며, 전형적인 {MBTI_TF} 유형의 사람처럼 대답합니다. 당신은 제가 하는 말에 {MBTI_TF}처럼 반응하거나, 질문에 대답하시면 됩니다. 반말만 사용하고, 한국어로만 대답하세요."
    start_index = START_IDX
    end_index = END_IDX

    with open(DATASET_PATH, "r") as r:
        dataset = json.load(r)

    if os.path.isfile(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r") as r:
            output = json.load(r)
            start_index = output[-1]['idx'] + 1

    # 리소스 이슈로 인해 start_index ~ end_index 개의 대화에만 진행
    for idx in range(start_index, end_index):
        try:
            prev_query = dataset[idx - 2]['query']
            prev_response = dataset[idx - 1]['query']
            cur_query = dataset[idx]['query']
            cur_response = dataset[idx]['response']

            messages = [
                {"role": "system", "content": chatgpt_prompt},
                {"role": "user", "content": prev_query},
                {"role": "assistant", "content": prev_response},
                {"role": "user", "content": cur_query}
            ]

            chatgpt_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=64
            )['choices'][0]['message']['content']

            conversation = {
                "idx": idx,
                "query": cur_query,
                "response": chatgpt_response
            }

            output.append(conversation)

            # Chatgpt API가 불안정해서 100번째 대화마다 저장
            if idx % 100 == 0:
                with open(OUTPUT_PATH, "w", encoding='utf-8') as wr:
                    json.dump(output, wr, indent=4, ensure_ascii=False)
                    print(f"{idx}'s conversation done")

        except Exception as e:
            print(e)
            time.sleep(10)
            continue
