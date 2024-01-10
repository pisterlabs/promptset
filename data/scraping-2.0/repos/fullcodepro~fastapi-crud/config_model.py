

from langchain import HuggingFaceHub, PromptTemplate, LLMChain
import os
from dotenv import load_dotenv
# Cargar las variables de entorno desde el archivo .env
load_dotenv() 

huggingfacehub_api_token = os.environ['HUGGINGFACEHUB_API_TOKEN']
repo_id = os.environ['REPO_ID']
llm = HuggingFaceHub(huggingfacehub_api_token=huggingfacehub_api_token,
                     repo_id=repo_id,
                     model_kwargs={"temperature": 0.6, "max_new_tokens": 1024 })

template="""{question}"""
template2 = """
You are an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.

{question}

"""



async def factory(question):
    print({question})
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)
    respuesta = llm_chain.run(question=question)
    print(respuesta)
    return respuesta

# result = factory("What is the meaning of life?")
# print(result)