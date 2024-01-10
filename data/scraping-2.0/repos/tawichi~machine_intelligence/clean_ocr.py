import openai
import os
from tqdm import tqdm

openai.api_key = "YOUR_API_KEY"

def get_subfolders(folder_path):
    subfolders = []
    for root, dirs, files in os.walk(folder_path):
        for dir in dirs:
            subfolder_path = os.path.join(root, dir)
            subfolders.append(subfolder_path)
    return subfolders

def get_completion(prompt, model="gpt-4"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


folder_path = "./sample"
subfolders = get_subfolders(folder_path)
for subfolder in tqdm(subfolders):
    with open(f'{subfolder}/raw_transcription.txt', 'r', encoding='utf-8') as file:
        content = file.read()
    

    # ここにプロンプト
    prompt = f"以下のバッククォーテーションで囲まれた文字列はあるウェブサイトのスクリーンショットからOCRによって得られたものである．この文字列の中から不要な空白・空行を取り除き，言語として意味の通じるように修復・整形せよ．なお，文字列から意味のある情報を失わないようにすること．修復・整形結果のみを出力すること．```{content}```"


    # GPT-4に入力
    response = get_completion(prompt)
    print(response)


    with open(f"{subfolder}/cleaned_transcription.txt", "w", encoding="utf-8") as file:
        file.write(response)