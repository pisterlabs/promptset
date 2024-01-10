from itertools import product
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

chat = ChatOpenAI(
    model="gpt-3.5-turbo",
)

prompt = PromptTemplate(
    template="{product}はどこの会社が開発した製品ですか？",
    input_variables=["product"],
)

chain = LLMChain(
    llm=chat,
    prompt=prompt,
    verbose=True,
)

result = chain.predict(product="iPhone")

print(result)