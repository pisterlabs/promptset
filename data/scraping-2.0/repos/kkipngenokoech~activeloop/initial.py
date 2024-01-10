from langchain.llms import OpenAI


llm = OpenAI(model="text-davinci-003", temperature=0.9)
prompt = "Once upon a time, there was a princess named Sophia who lived in..."
answer = llm(prompt)
print(answer)