from langchain.llms import OpenAI
# from langchain.llms import OpenAI

# 创建一个 LLM 实例
llm = OpenAI()
res =llm.predict("给我一个好听的 AI 创业公司的名称", temperature=0.5) # temperature 控制生成文本的多样性
print(res)

