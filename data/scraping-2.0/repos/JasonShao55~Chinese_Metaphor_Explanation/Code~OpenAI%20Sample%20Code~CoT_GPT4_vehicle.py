import json
import os
import openai
from tqdm import tqdm
import time


openai.api_type = "azure"
openai.api_version = "2023-05-15"
openai.api_base = "https://conmmunity-openai.openai.azure.com/"
# openai.api_key = os.getenv("OPENAI_API_KEY")  # 确保你已经设置了环境变量 OPENAI_API_KEY
openai.api_key = 'XXXX_YOUR_KEY'

# 载入json数据
data_path = r"D:\OneDrive\Learning\NLP Metaphor Dataset\data"

cluster_method = "embeddings"
model_engine = "gpt-4-0"

if cluster_method == "embeddings":
    initial_input_file = 'train_clustered_sampled_embeddings.json'
    output_file = model_engine + '-vehicle-embeddings-results.json'
    output_file_without_COT = model_engine + '-vehicle-results-no-COT.json'
elif cluster_method == "bert":
    initial_input_file = 'train_clustered_sampled.json'
    output_file = model_engine + '-vehicle-bert-results.json'
    output_file_without_COT = model_engine + '-vehicle-results-no-COT.json'

def prompt_with_ground():
    with open(os.path.join(data_path, initial_input_file), 'r', encoding='utf-8') as f:
        metaphors = json.load(f)

    # 初始化两个结果字典列表
    results = []

    # 构建初始的共性列表
    grounds = ['以' + m['tenor'] + '作为本体（tenor）' + "和以" + m[
        'ground'] + "作为共性（ground）构思隐喻时，对应的喻体（vehicle）可以是：" + m['vehicle'] for m in metaphors]
    initial_prompt = "在写作中，想构思隐喻（metaphor）时，有本体（tenor）和喻体（vehicle）的共性（ground）时就很容易构思对应的喻体，诸如由下述本体和共性构思隐喻时，它们的喻体可以是以下这些：" + "，".join(
        grounds)

    # 载入新的json数据
    with open(os.path.join(data_path, 'train_data_sampled_200.json'), 'r', encoding='utf-8') as f:
        new_metaphors = json.load(f)

    for new_metaphor in tqdm(new_metaphors, total=len(new_metaphors)):
        tenor = new_metaphor['tenor']
        ground = new_metaphor['ground']

        vehicle = None
        while vehicle is None:
            prompt = "请参考我给你展示的例子的内容和格式，直接给出" + tenor + '作为本体（tenor）和' + ground + "作为共性（ground）时的喻体（vehicle），不需要多加任何其他的解释或说明。"
            response = openai.ChatCompletion.create(
                engine=model_engine,
                messages=[
                    {"role": "system", "content": initial_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=800,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None
            )
            # 检查是否存在'content'键
            if 'content' in response.choices[0]['message']:
                vehicle = response.choices[0]['message']['content'].strip()
            else:
                print(f"No 'content' in response for {tenor} and {ground}")
                print(response.choices[0]['message'])
                # 没有得到有效的响应，继续循环

        metaphor = None
        while metaphor is None:
            prompt = "参考共性" + ground + "，并基于你给出的喻体：“" + vehicle + "”，写一个简短准确的含有关于" + tenor + "和" + vehicle + "的隐喻（metaphor）的中文句子。"
            response = openai.ChatCompletion.create(
                engine=model_engine,
                messages=[
                    {"role": "system", "content": initial_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=800,
                top_p=0.95,
                frequency_penalty=0,
                presence_penalty=0,
                stop=None
            )
            # 检查是否存在'content'键
            if 'content' in response.choices[0]['message']:
                metaphor = response.choices[0]['message']['content'].strip()
            else:
                print(f"No 'content' in response for {tenor} and {ground}")
                print(response.choices[0]['message'])
                # 没有得到有效的响应，继续循环

        results.append({"tenor": tenor, "ground": ground, "vehicle": vehicle, "metaphor": metaphor})
        # 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4, separators=(",", ": "))



def prompt_without_COT():
    # 初始化两个结果字典列表
    results = []

    # 载入新的json数据
    with open(os.path.join(data_path, 'train_data_sampled_200.json'), 'r', encoding='utf-8') as f:
        new_metaphors = json.load(f)

    for new_metaphor in tqdm(new_metaphors, total=len(new_metaphors)):
        tenor = new_metaphor['tenor']
        ground = new_metaphor['ground']
        id=new_metaphor['metaphor_id']

        metaphor = None
        while metaphor is None:
            prompt = "参考共性" + ground + "写一个简短准确的含有关于" + tenor+ "的隐喻（metaphor）的中文句子"
            for i in range(5):  # 尝试 5 次
                try:
                    response = openai.ChatCompletion.create(
                        engine=model_engine,
                        messages=[
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.7,
                        max_tokens=800,
                        top_p=0.95,
                        frequency_penalty=0,
                        presence_penalty=0,
                        stop=None,
                        timeout=1200
                    )
                    break  # 如果请求成功，跳出循环
                except openai.error.Timeout as e:
                    print("请求超时，正在重试...")
                    time.sleep(5)  # 等待 10 秒再重试

            # 检查是否存在'content'键
            if 'content' in response.choices[0]['message']:
                metaphor = response.choices[0]['message']['content'].strip()
            else:
                print(f"No 'content' in response for {id} and {tenor} and {ground}")
                print(response.choices[0]['message'])
                # 没有得到有效的响应，继续循环
            time.sleep(1)  # sleep for 5 seconds

        results.append({"tenor": tenor, "ground": ground, "metaphor": metaphor})
        # 保存结果
        with open(output_file_without_COT, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4, separators=(",", ": "))


if __name__ == '__main__':
    prompt_with_ground()
    prompt_without_COT()