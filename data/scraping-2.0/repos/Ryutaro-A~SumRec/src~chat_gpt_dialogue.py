import glob
import json
import openai
import os
import time
from tqdm import tqdm


openai.api_key = ""

def open_json(file_path, split_id):
    with open(file_path, mode="r", encoding="utf-8") as f:
        return [os.path.basename(data).replace(".json", "").replace(".rmd", "") for data in json.load(f)[split_id]["test"]]


def generate(prompt):
    time.sleep(1)
    while True:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k-0613",
                messages=[
                    {
                        "role": "user",
                        "content":prompt
                    },
                ],
                temperature=0.0,
            )
            break
        except:
            print('error')
            time.sleep(30)
            continue

    return response["choices"][0]["message"]["content"]


row_test_files = [
    # open_json("./data/crossval_split_5/unseen/all_topic_split.json", 3),
    # open_json("./data/crossval_split_5/unseen/only_trip_split.json", 1),
    open_json("./data/crossval_split_5/unseen/no_trip_split.json", 1),
]

data_type_list = [
    # 'all_topic',
    # 'only_trip',
    'no_trip'
]
data_dir = './data/GPT35sum_rec_chat_and_rec/'
out_dir = "./outputs/result/chatgpt_pre/dialogue/"



out_files = [os.path.basename(data).replace(".rmd.json", ".json") for data in glob.glob(out_dir + '*/*.json')]

for data_type, row_test in zip(data_type_list, row_test_files):
    with open(f'./data/prompt/dialogue-{data_type}.txt', encoding='utf-8') as f:
        base_prompt = f.read()
    files = glob.glob(data_dir + data_type + '/*.json')
    print("files:", len(files))
    test_files = [file_path for file_path in files if os.path.basename(file_path).replace(".json", "").replace(".rmd", "") in row_test]
    print("test:", len(test_files))
    for file_path in tqdm(test_files):
        filename = os.path.basename(file_path)

        # 既にあるファイルは飛ばす
        if filename in out_files:
            print('すでにあります')
            continue
        print(filename)
        with open(file_path, encoding='utf-8') as f:
                json_data = json.load(f)
        dialogue_list = ["A:"+data["utterance"] if i % 2 == 0 else "B:"+data["utterance"] for i, data in enumerate(json_data['dialogue'])]

        speakers = list(json_data["questionnaire"].keys())
        results = []
        for i, place_dict in enumerate(json_data["place"]):
            prompt = base_prompt \
                + "\n\n\n--6--\n" \
                + '【対話履歴】\n' \
                + "\n".join(dialogue_list)+'\n\n' \
                + '【観光地の説明】\n' \
                + place_dict["description"] + '\n\n' \
                + '【スコア】\n' \
                + 'Aさん:\n'

            sys_mes = generate(prompt)
            # print(prompt)
            print(sys_mes)
            results.append(sys_mes)

        print(file_path)
        os.makedirs(out_dir + data_type, exist_ok=True)
        with open(out_dir + data_type + '/' + os.path.basename(file_path).replace(".json", ".txt"), mode='w', encoding='utf-8') as f:
            f.write("\n\n".join(results))

        print()
        # exit()
