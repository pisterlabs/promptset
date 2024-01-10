import openai
import os

openai.api_key = "APIキーを指定"
#os.environ["OPENAI_API_KEY"] = "APIキーを指定"
openai.api_key = os.environ["OPEN_AI_API_KEY"]

def get_subfolders(folder_path):
    subfolders = []
    for root, dirs, files in os.walk(folder_path):
        for dir in dirs:
            subfolder_path = os.path.join(root, dir)
            subfolders.append(subfolder_path)
    return subfolders

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


folder_path = "./sample"
subfolders = get_subfolders(folder_path)
subfolders.sort()
#/でsplitした最後の要素をintとして取得し、それが51以上の場合に以下を実行
for subfolder in subfolders:
    if int(subfolder.split("/")[-1]) < 51:
        continue
    with open(f'{subfolder}/raw_transcription.txt', 'r', encoding='utf-8') as file:
        content = file.read()
    

    # ここにプロンプト
    prompt = f"Please clean up the following OCR transcribed documents by removing unnecessary blank spaces and blank lines. [{content}]"


    # GPT-4に入力
    response = get_completion(prompt)
    print(response)


    with open(f"{subfolder}/cleaned_transcription.txt", "w", encoding="utf-8") as file:
        file.write(response)