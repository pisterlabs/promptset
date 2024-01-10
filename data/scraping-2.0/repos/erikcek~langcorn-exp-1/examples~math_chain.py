import os

from langchain import LLMMathChain, OpenAI

os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "sk-********")

llm = OpenAI(temperature=0)
chain = LLMMathChain(llm=llm, verbose=True)
