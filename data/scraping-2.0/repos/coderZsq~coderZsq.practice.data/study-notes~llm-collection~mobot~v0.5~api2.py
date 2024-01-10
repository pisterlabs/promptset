from fastapi import FastAPI
import uvicorn

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

from langserve import add_routes


app = FastAPI(title="出生日期查询")

llm = ChatOpenAI(
    model_name='gpt-3.5-turbo-0613',
    temperature=0, max_tokens=2048,
    openai_api_key='replace to your api key'
)

prompt_template = "请回答 {name} 是什么时候出生的"

chain = LLMChain(llm=llm, prompt=PromptTemplate.from_template(prompt_template))

add_routes(app, chain)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5280)
