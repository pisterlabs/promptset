from langchain.llms import OpenAI
from langchain.llms.loading import load_llm

# 加载 OpenAI LLM 配置文件
llm = load_llm("./files/llm.json")
llm = load_llm("./files/llm.yaml")

# 保存 OpenAI LLM 配置文件
llm.save("./files/llm-1.json")
llm.save("./files/llm-1.yaml")

print(llm.predict("Tell me a joke"))