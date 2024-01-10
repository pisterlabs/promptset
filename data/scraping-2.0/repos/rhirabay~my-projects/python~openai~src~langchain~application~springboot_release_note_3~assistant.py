from llama_index import download_loader
from pathlib import Path
from llama_index import LLMPredictor, GPTVectorStoreIndex, PromptHelper, ServiceContext
from llama_index import QuestionAnswerPrompt, GPTVectorStoreIndex, SimpleDirectoryReader
from langchain import OpenAI
import os

urls=[
    'https://github.com/spring-projects/spring-boot/wiki/Spring-Boot-3.0-Migration-Guide'
]

# 環境変数の読み込み
# from dotenv import load_dotenv
# load_dotenv()

BeautifulSoupWebReader = download_loader("BeautifulSoupWebReader")

loader = BeautifulSoupWebReader()
documents = loader.load_data(urls=urls)

# index = GPTVectorStoreIndex.from_documents(documents)
# query_engine = index.as_query_engine()
# result = query_engine.query("what framework's migration guide?")

# モデルの定義 langchainを使う
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="gpt-3.5-turbo"))

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor)

index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

# ----- プロンプト -----
# QA_PROMPT_TMPL = (
#     "下記の情報が与えられています。 \n"
#     "---------------------\n"
#     "{context_str}"
#     "\n---------------------\n"
#     "この情報を参照して次の質問に答えてください: {query_str}\n"
# )
# QA_PROMPT = QuestionAnswerPrompt(QA_PROMPT_TMPL)

engine = index.as_query_engine()

response = engine.query("What should be done to migrate to Spring Boot 3.0?")
print(response.response)
