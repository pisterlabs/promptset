import os
from dotenv import load_dotenv
# 使用するクラスを明確化するために分割import
from langchain import PromptTemplate
from langchain.llms import OpenAI

# .envファイルからAPIキーを読み込む
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

template = """{question}
Use the following pieces of context to answer the users question.
Take note of the sources and include them in the answer in the format: "SOURCES: source1 source2", use "SOURCES" in capital letters regardless of the number of sources.
If you don't know the answer, just say that "I don't know", don't try to make up an answer.
"""

prompt = PromptTemplate(
    input_variables=["question"],
    template=template
)

# ChatGPTを使用、GPT3の場合はmodelname="text-davinci-003"
chatGPT = OpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=256, openai_api_key=OPENAI_API_KEY)

def generator(question):
    return chatGPT(prompt.format(question=question))