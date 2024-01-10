from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage
from langchain.chains.llm import LLMChain
from langchain.schema.document import Document
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
# ----------------------------------------------------------------------------------------------------------------------
from LLM2 import llm_config
from LLM2 import llm_models
from LLM2 import llm_chains
from LLM2 import llm_Asistant_OPENAI
# ----------------------------------------------------------------------------------------------------------------------
def ex1_openai_chat(prompt='Who framed Roger rabit?'):
    # OK
    llm_cnfg = llm_config.get_config_openAI()
    LLM = llm_models.get_model(llm_cnfg.filename_config_chat_model,model_type='QA')
    chain = llm_chains.get_chain_chat(LLM)
    response = chain.run({'question':prompt, 'input_documents':[]})
    print(response)
    print(''.join(['-'] * 50))
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex2_openai_summarize(prompt='What is common across the texts.Response with single word.',texts=['Cofee','Milk','Beer','Wine','Water']):

    llm_cnfg = llm_config.get_config_openAI()
    LLM = llm_models.get_model(llm_cnfg.filename_config_chat_model, model_type='QA')

    map_template = """The following is a set of documents {docs} Based on this list of docs, please identify the main themes Helpful Answer:"""
    map_chain = LLMChain(llm=LLM, prompt=PromptTemplate.from_template(map_template))
    res = map_chain.run(prompt + ','.join(texts))

    print(res)
    print(''.join(['-'] * 50))

    return
# ----------------------------------------------------------------------------------------------------------------------
def ex3_Azure_chat(prompt='Who framed Roger rabit?'):
    #OK
    llm_cnfg = llm_config.get_config_azure()
    LLM = llm_models.get_model(llm_cnfg.filename_config_chat_model, model_type='QA')
    res = LLM([HumanMessage(content=prompt)]).content
    print(res)
    print(''.join(['-'] * 50))
    return
# ----------------------------------------------------------------------------------------------------------------------
def ex4_Azure_summarize(prompt='What is common across the texts.Response with single word:',texts=['Cofee','Milk','Beer','Wine','Water']):

    llm_cnfg = llm_config.get_config_azure()
    LLM = llm_models.get_model(llm_cnfg.filename_config_chat_model, model_type='Summary')
    docs = [Document(page_content=t, metadata={}) for t in texts]
    prompt_template = """%s {text}"""%prompt
    prompt = PromptTemplate.from_template(prompt_template)
    stuff_chain = StuffDocumentsChain(llm_chain=LLMChain(llm=LLM, prompt=prompt), document_variable_name="text")
    res = stuff_chain.run(docs)
    print(res)
    print(''.join(['-']*50))
    return
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    #ex1_openai_chat()
    #ex2_openai_summarize()
    #ex3_Azure_chat()
    #ex4_Azure_summarize()
    from LLM2 import llm_RAG

    llm_cnfg = llm_config.get_config_azure()
    LLM = llm_models.get_model(llm_cnfg.filename_config_chat_model, model_type='QA')
    chain = llm_chains.get_chain_chat(LLM)
    res = chain.run(question='1+1', input_documents=[])
    print(res)