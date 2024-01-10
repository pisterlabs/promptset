from langchain.chains import LLMChain, LLMRequestsChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

chat = ChatOpenAI()

prompt = PromptTemplate(
    input_variables=["query", "requests_result"],
    template="""
        以下の文章を元に質問に答えてください。
        文章: {requests_result}
        質問: {query}
        """
)

llm_chain = LLMChain(
    llm=chat,
    prompt=prompt,
    verbose=True,
)

chain = LLMRequestsChain(
    llm_chain=llm_chain,
    verbose=True,
)

print(chain({
    "query": "明日の東京は寒いですか？",
    "url": "https://www.jma.go.jp/bosai/forecast/data/overview_forecast/130000.json",
}))