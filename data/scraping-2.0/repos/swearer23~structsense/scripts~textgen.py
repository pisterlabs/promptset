import json
import requests
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import (
    PromptTemplate,
)

extract_schema = {
  "renter": "房屋出租方姓名",
  "rentee": "房屋承租方姓名",
}

extract_template = json.dumps(extract_schema)
example_response = json.dumps({"renter": "张三"})

prompt = PromptTemplate(
    template="""
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
从以下Input部分的内容中抽取以下信息：
```
{extract_template}
```
response部分中应仅包含抽取信息的json格式，不要添加任何评论

### Input:
{query}

### Response:

    """,
    input_variables=["query", "extract_template"]
)


loader = PyPDFLoader('docs/lianjia.pdf')
text_splitter = CharacterTextSplitter(
    chunk_size = 5000,
    chunk_overlap  = 200,
    length_function = len,
    is_separator_regex = False,
)
pages = loader.load_and_split(text_splitter)

inp = prompt.format_prompt(query=pages[1].page_content.replace('\n', '\\n').replace('\t', '\\t'), extract_template=extract_template)

url = "http://127.0.0.1:5000/v1/chat/completions"

headers = {
    "Content-Type": "application/json"
}

data = {
    "mode": "completion",
    "character": "Example",
    "messages": inp.text
}
response = requests.post(url, headers=headers, json=data, verify=False)
assistant_message = response.json() #['choices'][0]['message']['content']
print(assistant_message)
