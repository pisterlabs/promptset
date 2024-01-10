import openai
import os
import time 

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
for subfolder in subfolders:
    if int(subfolder.split("/")[-1]) < 63:
        continue
    with open(f'{subfolder}/raw_transcription.txt', 'r', encoding='utf-8') as file:
        content = file.read()
    

    # ここにプロンプト
    prompt = f"[{content}] The text above is an OCR transcription of the screenshot. From this information only, please guess what the original screenshot was taken of and output four candidates. Example) #Website#,#Code Editor#,#Blog# Please output the guessed words surrounded by #."

    # GPT-4に入力
    response = get_completion(prompt)
    print(response)

    # ##で囲われた単語を配列に格納
    extracted = []
    for word in response.split("#"):
        if word != "" and word != "," and word != " ":
            extracted.append(word)

    #　改行で区切り、一つの文字列にする
    response = "\n".join(extracted)        

    with open(f"{subfolder}/predicted_label.txt", "w", encoding="utf-8") as file:
        file.write(response)
    
    time.sleep(0.3)