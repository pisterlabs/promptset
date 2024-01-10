# load env
from dotenv import load_dotenv
load_dotenv(dotenv_path="../../../../.env")

# load index
from llama_index import StorageContext, load_index_from_storage
storage_context = StorageContext.from_defaults(persist_dir="../../.data")
index = load_index_from_storage(storage_context)

# imports
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from llama_index.prompts import Prompt

template_messages = [
    SystemMessagePromptTemplate.from_template(
        "与えられたドキュメントから回答ができない場合、「与えられたドキュメントでは回答できません」と回答してください。"
    ),
    HumanMessagePromptTemplate.from_template("""
        ドキュメントはこちらです。
        ---------------------
        {context_str}
        ---------------------
        ドキュメントの内容をもとに以下の質問に回答してください。
        質問: {query_str}
    """),
]

templates = ChatPromptTemplate.from_messages(template_messages)
text_qa_template = Prompt.from_langchain_prompt(templates)

query_engine = index.as_query_engine(text_qa_template=text_qa_template)
query = "ジョージの生い立ちを教えて"
print(f'query: {query}')
response = query_engine.query(query)
print(f'response: {response}')

query = "アンナの生い立ちを教えて"
print(f'query: {query}')
response = query_engine.query(query)
print(f'response: {response}')
