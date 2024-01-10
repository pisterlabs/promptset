#!/usr/bin/env python3
# -*- coding utf-8 -*-
import yaml
from langchain import PromptTemplate, HuggingFaceHub, LLMChain

'''
用于演示如何将HuggingFace Hub和LangChain集成
'''

def get_hf_key():
    with open("config.yaml", "r", encoding="utf-8") as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)
        api_key = yaml_data["huggingface"]["api_key"]
        return api_key

if __name__ == '__main__':
    # 导入HuggingFace API Token
    get_hf_key()

    # 初始化HF LLM
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-small",
        #repo_id="meta-llama/Llama-2-7b-chat-hf",
    )

    # 创建简单的question-answering提示模板
    template = """Question: {question}
                  Answer: """

    # 创建Prompt        
    prompt = PromptTemplate(template=template, input_variables=["question"])

    # 调用LLM Chain
    llm_chain = LLMChain(
        prompt=prompt,
        llm=llm
    )

    # 准备问题
    question = "Rose is which type of flower?"

    # 调用模型并返回结果
    print(llm_chain.run(question))
