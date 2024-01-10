from langchain import HuggingFaceHub, LLMChain, PromptTemplate

# 使用Baichuan-7B预训练模型
repo_id = "baichuan-inc/Baichuan-7B"
llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature": 0, "max_length": 64})

# 创建prompt预定义模板
template = """Question: {question}
Answer: Let's think step by step."""
prompt = PromptTemplate(template=template, input_variables=["question"])

# 创建LLMChain对象
llm_chain = LLMChain(prompt=prompt, llm=llm)

# 运行并生成文本
question = "Who won the FIFA World Cup in the year 1994? "
print(llm_chain.run(question))

print("done!")
