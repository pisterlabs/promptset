

#write fan fictions based on the instruction and db as background
def run(instruction:str):
    #input: the instruction for fan fiction
    #output: fiction based on the instruction
    from langchain.chains.llm import LLMChain
    from langchain.prompts import PromptTemplate
    from langchain.chains.combine_documents.stuff import StuffDocumentsChain
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.chat_models import ChatOpenAI

    db = Chroma(persist_directory="./db/arkdb", embedding_function=OpenAIEmbeddings())
    res = db.similarity_search(instruction, k = 6)
    
    # Define prompt
    prompt_template = "based on the text as the background, " + instruction + """ 
    "{text}"
    Output:"""
    prompt = PromptTemplate.from_template(prompt_template)
    
    # Define LLM chain
    llm = ChatOpenAI(temperature=0.8, model_name="gpt-4")
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    
    # Define StuffDocumentsChain
    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain, document_variable_name="text"
    )
    return(stuff_chain.run(res))



#call fiction by api, allows NSFW local model to write fictions
import requests
#Chat generate
HOST = 'localhost:5000'
URI = f"http://{HOST}/api/v1/generate"
def request(context, prompt):
    body = {
        "prompt": context + '\n Based on the context above,' + prompt + '\n: Output:',
        "max_new_tokens": 2050,
        "do_sample": True,
        "temperature": 0.65,
        "top_p": 0.9,
        "typical_p": 1,
        "repetition_penalty": 1.18,
        "top_k": 0,
        "min_length": 10,
        "no_repeat_ngram_size": 0,
        "num_beams": 1,
        "penalty_alpha": 0,
        "length_penalty": 1,
        "early_stopping": False,
        "seed": -1,
        "add_bos_token": True,
        "truncation_length": 2048,
        "ban_eos_token": False,
        "skip_special_tokens": True,
        "stopping_strings": [],
    }
    response = requests.post(URI, json=body, timeout=120)
    return response.json()


#write NSFW fan fictions based on the instruction and db as background
#use api request to connect to local model for NSFW contents
def run_api(context:str, instruction:str):
    #use the gpt chain as the summary chain
    #context = run(instruction)
    #re pass the result as the context to api call to the local model
    response = request(context, instruction)
    return(response['results'][-1]['text'])

