from dotenv import load_dotenv
import os
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from pprint import pprint

load_dotenv()

schema = {
    "properties": {
        "name": {"type": "string"},
        "height": {"type": "integer"},
        "hair_color": {"type": "string"},
    },
    "required": ["name", "height"],
}

contract_schema = {
    "properties": {
      "房屋出租方姓名": {"type": "string"},
      "房屋出租方性别": {"type": "string"},
      "房屋出租方身份证号码": {"type": "string"},
      "房屋承租方姓名": {"type": "string"},
      "房屋承租方性别": {"type": "string"},
      "房屋承租方身份证号码": {"type": "string"},
      "出租房屋坐落地址": {"type": "string"},
      "原合同编号": {"type": "string"},
      "本合同编号": {"type": "string"},
      "建筑面积": {"type": "string"},
      "每月租金": {"type": "string"},
    },
    "required": ["房屋出租方", "房屋承租方"],
}

template = '''
You are a PDF parser expert to extract assigned content from given content or file.
Answer the user query based on instruction and query content.
Based on the following contract schema, extract the content from the given contract.

# input:

{input}

# now extract all tables from the given contract and reply the extracted content in json format.:

'''

loader = PyPDFLoader('./docs/PO1077867-0.PDF')
text_splitter = CharacterTextSplitter(
    chunk_size = 5000,
    chunk_overlap  = 200,
    length_function = len,
    is_separator_regex = False,
)
pages = loader.load_and_split(text_splitter)
text = '\n'.join([page.page_content for page in pages])

chat_prompt = PromptTemplate.from_template(template)

chat_prompt = chat_prompt.format(input=text)

# This code is for v1 of the openai package: pypi.org/project/openai
from openai import OpenAI
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

response = client.chat.completions.create(
  model="gpt-3.5-turbo-16k",
  messages=[{
    "role": "assistant",
    "content": chat_prompt,
  }],
  temperature=0.2,
)

pprint(response)
