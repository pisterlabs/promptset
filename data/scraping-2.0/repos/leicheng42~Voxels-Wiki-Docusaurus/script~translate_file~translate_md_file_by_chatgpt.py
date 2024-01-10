import os
from revChatGPT.V1 import Chatbot
import yaml
import openai

# 读取YAML文件
with open('script/translate_file/config.yaml', 'r') as yaml_file:
    data = yaml.safe_load(yaml_file)

# 读取配置
access_token = data['access_token_list'][1]
BASE_URL = data['BASE_URL'][1]

# openai.api_key = "这里填 access token，不是 api key"
openai.api_base = BASE_URL
openai.api_key = access_token

# 递归获取文件夹中的所有Markdown文件
def find_markdown_files(folder):
    markdown_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.endswith('.md'):
                markdown_files.append(os.path.join(root, file))
    return markdown_files

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


def translate_by_chatgpt(content):
    """
    由chatgpt翻译
    """
    prompt = "请帮我翻译以下文本（不要输出额外的提示）：\n" + content

    print("prompt:\n", prompt)

    print("开始翻译！")

    response = get_completion(prompt, model="gpt-3.5-turbo-16k-0613")

    
    print("response:\n", response)

    # # create variables to collect the stream of events
    # completion_text = ''

    # response = openai.ChatCompletion.create(
    #     model='gpt-3.5-turbo',
    #     messages=[
    #         {'role': 'user', 'content': prompt},
    #     ],
    #     stream=True,
    #     allow_fallback=True,
    #     temperature=0,
    # )

    # for chunk in response:
    #     chunk_text = chunk.choices[0].delta.get("content", "")  # extract the text
    #     completion_text += chunk_text  # append the text

    #     # 将换行符替换为空格，避免意外换行
    #     print_completion_text = completion_text[-50:].replace("\n", " ")

    #     print(print_completion_text + " "*15, end="\r", flush=True)

    # print("\n")

    return response


# 递归查找Markdown文件
markdown_files = find_markdown_files('docs/Player_customization')

# markdown_files = ["docs/home.md"]
# markdown_files = ["docs/features/submit-library.md"]

base_folder = "docs_translate"

# 下载每个Markdown文件中的资源
# for markdown_file in markdown_files:
for index in range(len(markdown_files)):
    markdown_file = markdown_files[index]
    with open(markdown_file, 'r', encoding='utf-8') as file:
        markdown_content = file.read()

    print("翻译进度：{} / {}".format(index + 1, len(markdown_files)))

    translate_content = translate_by_chatgpt(markdown_content)

    # 提取文件夹路径（不包括文件名）
    folder_path = os.path.join(base_folder, os.path.dirname(markdown_file))
    os.makedirs(folder_path, exist_ok=True)

    # 将修改后的内容写回到原始文件
    with open(os.path.join(base_folder, markdown_file), 'w', encoding='utf-8') as file:
        file.write(translate_content)
    
    print(f'Translate Markdown content saved to {markdown_file}')

print("翻译完成！")