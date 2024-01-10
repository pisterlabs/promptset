from langchain.llms.openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo-instruct")

result = llm(
    "おいしいラーメンが",
    stop="。",
)

print(result)
