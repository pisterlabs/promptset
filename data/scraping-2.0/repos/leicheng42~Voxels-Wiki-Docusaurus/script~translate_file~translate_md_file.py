import os
from revChatGPT.V1 import Chatbot
import yaml
import openai
from transformers import GPT2Tokenizer

# OpenAI GPT-2 tokenizer is the same as GPT-3 tokenizer
# we use it to count the number of tokens in the text
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# 读取YAML文件
with open('script/translate_file/config.yaml', 'r') as yaml_file:
    data = yaml.safe_load(yaml_file)

# 读取配置
access_token = data['access_token_list'][3]
BASE_URL = data['BASE_URL'][2]

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

    response = get_completion(prompt, model="gpt-3.5-turbo")


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


def group_chunks(chunks, ntokens, max_len=1000, hard_max_len=3000):
    """
    Group very short chunks, to form approximately page long chunks.
    """
    batches = []
    cur_batch = ""
    cur_tokens = 0
    
    # iterate over chunks, and group the short ones together
    for chunk, ntoken in zip(chunks, ntokens):
        # discard chunks that exceed hard max length
        if ntoken > hard_max_len:
            print(f"Warning: Chunk discarded for being too long ({ntoken} tokens > {hard_max_len} token limit). Preview: '{chunk[:50]}...'")
            continue

        # if room in current batch, add new chunk
        if cur_tokens + 1 + ntoken <= max_len:
            cur_batch += "\n\n" + chunk
            cur_tokens += 1 + ntoken  # adds 1 token for the two newlines
        # otherwise, record the batch and start a new one
        else:
            batches.append(cur_batch)
            cur_batch = chunk
            cur_tokens = ntoken
            
    if cur_batch:  # add the last batch if it's not empty
        batches.append(cur_batch)
        
    return batches


def translate_markdown_content(markdown_content):
    """
    translate_markdown_file
    split
    group
    translate
    concate
    """

    # split markdown content

    chunks = markdown_content.split('\n\n')

    ntokens = []
    for chunk in chunks:
        ntokens.append(len(tokenizer.encode(chunk)))

    chunks = group_chunks(chunks, ntokens)

    translated_chunks = []
    for i, chunk in enumerate(chunks):
        print(str(i+1) + " / " + str(len(chunks)))
        # translate each chunk
        translated_chunks.append(translate_by_chatgpt(markdown_content))

    # join the chunks together
    translated_markdown_content = '\n\n'.join(translated_chunks)

    return translated_markdown_content


if __name__ == "__main__":

    # 递归查找Markdown文件
    # markdown_files = find_markdown_files('docs/Player_customization')

    markdown_files = ["docs/home.md"]
    # markdown_files = ["docs/features/submit-library.md"]

    base_folder = "docs_translate"

    # 下载每个Markdown文件中的资源
    # for markdown_file in markdown_files:
    for index in range(len(markdown_files)):
        markdown_file = markdown_files[index]
        with open(markdown_file, 'r', encoding='utf-8') as file:
            markdown_content = file.read()

        print("翻译进度：{} / {}".format(index + 1, len(markdown_files)))

        translated_markdown_content = translate_markdown_content(markdown_content)

        # 提取文件夹路径（不包括文件名）
        folder_path = os.path.join(base_folder, os.path.dirname(markdown_file))
        os.makedirs(folder_path, exist_ok=True)

        # 将修改后的内容写回到原始文件
        with open(os.path.join(base_folder, markdown_file), 'w', encoding='utf-8') as file:
            file.write(translated_markdown_content)
        
        print(f'Translate Markdown content saved to {markdown_file}')

    print("翻译完成！")