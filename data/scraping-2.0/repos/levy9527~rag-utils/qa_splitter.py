import logging
import os
import sys
from datetime import datetime
from typing import List, AnyStr
import uuid
import hashlib
import argparse

import openai
import chromadb
from chromadb import Settings
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import tiktoken
import re

TRUE = 'true'

load_dotenv()
AZURE_API_VERSION = '2023-07-01-preview'
OPENAI_API_TYPE = 'azure'
OPENAI_API_BASE = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_API_KEY = os.getenv("AZURE_OPENAI_KEY")

MAX_TOKENS = 4096


def main():
  logging.basicConfig(level=logging.INFO)
  #print(os.environ)

  parser = argparse.ArgumentParser()
  # Add the positional argument for the filename (required)
  parser.add_argument("filename", help="markdown file to be split")
  # Add the optional argument for the delimiter
  parser.add_argument("--delimiter", help="Specify the delimiter string")

  args = parser.parse_args()
  filename = args.filename
  delimiter = args.delimiter

  if delimiter is None:
    print("delimiter is not specified. example: --delimiter=问题:")
    sys.exit(0)

  logging.info('opening file...')
  with open(filename, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    regrouped_lines = regroup_by_delimiter(lines, delimiter)

    # 针对QA，特殊处理
    is_keyword_search = TRUE
    answers = list(map(lambda x: '', range(len(regrouped_lines))))
    chunks = []

    if is_keyword_search == TRUE:
      logging.info('special chunking: is_keyword_search')
      chunks = list(map(lambda x: trim(x[0], delimiter), regrouped_lines))
      answers = list(map(lambda x: "".join(line for line in x[1:]), regrouped_lines))
    else:
      logging.info("common chunking")
      chunks = list(map(lambda x: "".join(line for line in x), regrouped_lines))

    client = get_chroma()
    collection = get_collection(client)

    # TODO need solution to work around this: what if exceed token limit?
    for index, chunk in enumerate(chunks):
      num = num_tokens_from_string(chunk)
      # check token
      if num > MAX_TOKENS:
        logging.info(f"strlen exceed token limit: {num}")
        sys.exit(1)
      else:

        logging.info('put data into chroma...')

        # 为什么用 uuid? 因为不能批量操作（数据太多），则必须考虑失败重试、重复插入的情况，此时 hash 生成的 id 是稳定的。
        # 当然，这也引入了 hash 冲突的风险，sha224 概率上足够了，如果冲突了，把无法插入的文本修改一下，再重新插入。
        collection.upsert(
          documents=[chunk],
          metadatas=[{"source": os.path.splitext(filename)[0],
                      'index': index,
                      'is_keyword_search': is_keyword_search,
                      'answer': answers[index],
                    }],
          ids=[get_hash(chunk)]
        )

    logging.info("job done!")

def trim(s, delimiter):
  '''
  remove delimiter and line feed
  '''
  sub = re.sub(delimiter, '', s)
  return sub.replace('\n', '').strip()


def num_tokens_from_string(string: str, encoding_name="cl100k_base") -> int:
  """Returns the number of tokens in a text string."""
  encoding = tiktoken.get_encoding(encoding_name)
  num_tokens = len(encoding.encode(string))
  logging.info(f"Token count: {num_tokens}")
  return num_tokens

def regroup_by_delimiter(lines: List[AnyStr], delimiter: str):
  logging.info('regroup_by_delimiter, {}'.format(delimiter))
  '''
  now only support split by line which startswith delimiter.
  以QA问答为例，返回一个二维数组[[QA1], [QA2]]。QA格式示例：[这是问题\n，这是答案的1行\n，这是答案的2行\n]
  '''
  result = []
  subgroup = []
  for line in lines:
    if line.startswith(delimiter):
      if subgroup:
        result.append(subgroup)
      subgroup = [line]
    else:
      subgroup.append(line)
  if subgroup:
    result.append(subgroup)
  return result


def get_embedding(text, model="text-embedding-ada-002"):
  logging.info('getting embeddings...')
  openai.api_type = OPENAI_API_TYPE
  openai.api_version = AZURE_API_VERSION
  openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")  # Your Azure OpenAI resource's endpoint value.
  openai.api_key = os.getenv("AZURE_OPENAI_KEY")

  text = text.replace("\n", " ")
  return openai.embeddings.create(input = [text], model=model).data[0].embedding

def get_chroma(host="10.201.0.32", port="8080"):
  return chromadb.HttpClient(host, port, settings=Settings(allow_reset=True))

def get_collection(client):
  metadata = {
    "create_by": "levy",
    "create_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
  }
  collection = client.get_or_create_collection('deinsight', metadata=metadata,
                                               embedding_function=embedding_functions.OpenAIEmbeddingFunction(
                                                 api_key=OPENAI_API_KEY,
                                                 api_base=OPENAI_API_BASE,
                                                 api_type="azure",
                                                 api_version=AZURE_API_VERSION,
                                                 model_name="text-embedding-ada-002")
                                               )
  logging.info(collection)
  return collection

def get_uuid():
  random_uuid = uuid.uuid4()
  return str(random_uuid)

def get_hash(content):
  hash_object = hashlib.sha224()

  # Convert the content to bytes and update the hash object
  hash_object.update(content.encode('utf-8'))

  # Get the hexadecimal representation of the hash
  hash_value = hash_object.hexdigest()

  return hash_value


if __name__ == '__main__':
  main()

