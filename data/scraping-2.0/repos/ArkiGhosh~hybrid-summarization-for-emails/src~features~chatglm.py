from langchain.chains import LLMChain
from langchain.llms.chatglm import ChatGLM
from langchain.prompts import PromptTemplate

import os

class GLM:

    def __init__(self, endpoint_url):
        self.template = """{question}"""
        self.prompt_template = PromptTemplate(template = self.template, input_variables = ["question"])
        self.endpoint_url = endpoint_url
        self.llm = ChatGLM(
            endpoint_url = self.endpoint_url,
            max_token = 80000,
            top_p=0.9,
            model_kwargs={"sample_model_args": False},
        )

    def summarize(self, prompt):
        llm_chain = LLMChain(prompt = self.prompt_template, llm = self.llm)
        return llm_chain.run(prompt)
    
a = GLM("http://127.0.0.1:4000")
a.summarize("What is the color of an apple?")

        
