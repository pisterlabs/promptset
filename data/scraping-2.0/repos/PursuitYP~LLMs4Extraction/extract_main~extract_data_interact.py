""" Part 1: initialize environment and declare utils functions """
import os
import openai
import textract
import tiktoken
import ast
import json
from dotenv import load_dotenv

load_dotenv()

os.environ["http_proxy"] = "http://10.10.1.3:10000"
os.environ["https_proxy"] = "http://10.10.1.3:10000"


# Split a text into smaller chunks of size n, preferably ending at the end of a sentence
def create_chunks(text, n, tokenizer):
    tokens = tokenizer.encode(text)
    """Yield successive n-sized chunks from text."""
    i = 0
    while i < len(tokens):
        # Find the nearest end of sentence within a range of 0.5 * n and 1.5 * n tokens
        j = min(i + int(1.5 * n), len(tokens))
        while j > i + int(0.5 * n):
            # Decode the tokens and check for full stop or newline
            chunk = tokenizer.decode(tokens[i:j])
            if chunk.endswith(".") or chunk.endswith("\n"):
                break
            j -= 1
        # If no end of sentence found, use n tokens as the chunk size
        if j == i + int(0.5 * n):
            j = min(i + n, len(tokens))
        yield tokens[i:j]
        i = j


# get completion with gpt-3.5-turbo
def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]


# Use text-davinci-003 with designed prompt to extract data
def extract_chunk_davinci(document,template_prompt):
    
    prompt=template_prompt.replace('<document>',document)

    response = openai.Completion.create(
        model='text-davinci-003', 
        prompt=prompt,
        temperature=0,
        max_tokens=1500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return "1." + response['choices'][0]['text']


# Use gpt-3.5-turbo with designed prompt to extract data
def extract_chunk(document, template_prompt):
    
    prompt=template_prompt.replace('<document>', document)

    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo', 
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        max_tokens=1500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return "1. " + response['choices'][0]['message']['content']
""" Part 1 end """


""" Part 2: upload and parse a .pdf file, and load the API-key """
# User upload a .pdf file
text = textract.process('data/radiolarian/715.pdf', method='pdfminer').decode('utf-8')
clean_text = text.replace("  ", " ").replace("\n", "; ").replace(';',' ')
# Initialise tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")

# Load API key from user input
# @施涛 @李沿澎: 用户输入API-key的字符串，填充到API-KEY字段
# 上一版本demo系统的API-KEY位置，保持原样
openai.api_key = " "    # load from "enter your API-key" field
os.environ['OPENAI_API_KEY'] = openai.api_key
""" Part 2 end """


""" Part 3: extract structured data with gpt-3.5-turbo """
# @李沿澎: 加一个文本说明框“抽取放射虫相关的结构化数据”，一个按钮“开始”，点击按钮执行下面代码
# 上一版demo系统的Command位置，按上述内容改一下，不需要用户输入了
document = '<document>'
template_prompt=f'''Extract key pieces of information from this regulation document.
If a particular piece of information is not present, output \"Not specified\".
When you extract a key piece of information, include the closest page number.
Use the following format:
0. What is the title
1. What is the section name
2. What are the locations of the boulders, samples and sections
3. What is the gps location
4. What are the associated fossils
5. What is the lithology
6. What is the number of species or genera found
7. What is the number of new species or new genera found

Document: \"\"\"{document}\"\"\"\n
0. Who is the title: radiolarian paper (Page 1)
1.'''

results = []
chunks = create_chunks(clean_text, 1000, tokenizer)
text_chunks = [tokenizer.decode(chunk) for chunk in chunks]
# 处理进度提示 - "系统流程: 关键信息提取->数据格式转换->结构化数据生成"
# 处理进度提示 - "Phase 1: 关键信息提取（转圈…）"
for chunk in text_chunks:
    results.append(extract_chunk(chunk, template_prompt))

groups = [r.split('\n') for r in results]
# zip the groups together
zipped = list(zip(*groups))
zipped = [x for y in zipped for x in y if "Not specified" not in x and "__" not in x]
# 处理进度提示 - "Phase 1: 关键信息提取（完成！）"


# 处理进度提示 - "Phase 2: 数据格式转换（转圈…）"
data_list = zipped
prompt = f"""
Translate the following python list to a dictionary with "section name", "locations of the boulders, samples and sections", \
    "gps location", "associated fossils", "lithology", "number of species or genera", and "number of new species or new genera" \
    as the keys, and set the values as precisely and concisely as possible: {data_list}
"""
# 处理进度提示 - "Phase 2: 数据格式转换（完成！）"


# 处理进度提示 - "Phase 3: 结构化数据生成（转圈…）"
response = get_completion(prompt)
res_json = ast.literal_eval(response)
res_json_out = json.dumps(res_json, indent=4)
# @李沿澎: 最后输出结构化的json格式数据展示就可以
print(res_json_out)
# 处理进度提示 - "Phase 3: 结构化数据生成（完成！）"
""" Part 3 end """
