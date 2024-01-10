import os
from langchain import HuggingFaceHub,LLMChain,HuggingFacePipeline
from langchain.llms import Replicate
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.llms import OpenAI



import re

def create_model(llm_name, llm_temp, llm_maxlen,openaikey):
    
    load_dotenv()
    os.environ["OPENAI_API_KEY"]=openaikey
    if re.search("^meta/llama", llm_name):
        if llm_name =="meta/llama-2-70b-chat":
            repo_name="meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3"
        elif llm_name =="meta/llama-2-13b-chat":
            repo_name="meta/llama-2-13b-chat:f4e2de70d66816a838a89eeeb621910adffb0dd0baba3976c96980970978018d"
        elif llm_name =="meta/llama-2-7b-chat":
            repo_name="meta/llama-2-7b-chat:13c3cdee13ee059ab779f0291d29054dab00a47dad8261375654de5540165fb0"
        return Replicate(
            model=repo_name,model_kwargs={"temperature": llm_temp, "max_length": llm_maxlen},)
    elif re.search("^GPT", llm_name):
        local_llm=OpenAI(temperature=0.1)
        prompt=PromptTemplate(input_variables=['userquest'],
        template = """you are a helpful assistant answering users' questions clearly. {userquest} """)
        llmchain=LLMChain(prompt =prompt, llm = local_llm, verbose= True)
        return llmchain
    else:
        hub_llm=HuggingFaceHub(repo_id=llm_name,model_kwargs={"temperature": llm_temp, "max_length": llm_maxlen})

        prompt=PromptTemplate(input_variables=['userquest'],
                        template = """you are a helpful assistant answering users' questions clearly. {userquest}  """)
        
        llm_chain = LLMChain(prompt=prompt, llm = hub_llm, verbose= True)    
        return llm_chain

def generate_answer(llm_chain, question,modelname):
    if re.search("^meta/llama", modelname):
        return llm_chain(f"""you are a helpful assistant answering users questions clearly. {question}  """)
    return llm_chain.run(question)

