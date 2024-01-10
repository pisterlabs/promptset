import json

import dotenv
import os
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts import PromptTemplate
import getArticle

dotenv_file = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv_file)

def searchArticleByUserKeyword(user_keyword):
    keyword_gen_template = """아래의 질문을 몇가지 키워드로 요약해줘.
    질문: {question}"""

    prompt = PromptTemplate.from_template(keyword_gen_template)
    prompt.format(question="아이유 몇살이야?")

    api_key = os.environ["OPENAI_API_CD"]
    chat_model = ChatOpenAI(openai_api_key=api_key)
    keyword = chat_model.predict(prompt.format(question=user_keyword))
    print(keyword)
    return keyword

# LLMs: this is a language model which takes a string as input and returns a string
llm = OpenAI(openai_api_key=os.environ["OPENAI_API_KEY"])
# ChatModels: this is a language model which takes a list of messages as input and returns a message
chat_model = ChatOpenAI(openai_api_key=os.environ["OPENAI_API_KEY"])

template = "당신은 주어진 articles를 기반으로 question을 답해야 합니다.\
            답할 수 있는 경우 답과 함께 근거 article를 붙여 서술하고,\
            알 수 없는 경우 '모르겠습니다.'라고 답변하세요."
human_template = "articles: {articles},\n" \
                  "question: {question_keyword}"

chat_prompt = ChatPromptTemplate.from_messages([
    ("system", template),
    ("human", human_template),
])

chain = chat_prompt | chat_model

if __name__ == "__main__":
    user_keyword = searchArticleByUserKeyword(input())
    user_keyword = user_keyword.replace(",", "")
    article_string = getArticle.getArticleDetailBulkWithStr(user_keyword)
    result = chain.invoke({"articles":article_string[0:3500],"question_keyword":user_keyword})
    print(result.content)
    # print(type(result.content)) #'str'
