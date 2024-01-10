'''
pip install gpt_index
pip install langchain
pip install transformers
'''

import os
import openai
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, LLMPredictor, PromptHelper, SimpleDirectoryReader
from langchain import OpenAI
from llama_index import GPTVectorStoreIndex, TrafilaturaWebReader
import chromadb
from mySecrets import LOCAL_PATH, OpenAI_API_KEY_PERSONAL

os.environ['OPENAI_API_KEY'] = OpenAI_API_KEY_PERSONAL
openai.api_key = OpenAI_API_KEY_PERSONAL
# openai.organization = ""
vIdx = LOCAL_PATH + 'data/source_of_knowledge/vectorIndex.json'



def createVectorIndex():

    max_input = 4096
    tokens = 256
    chunk_size = 600
    max_chunk_overlap = 0.5

    prompt_helper = PromptHelper(max_input, tokens, max_chunk_overlap, chunk_size_limit=chunk_size)

    #define LLM
    llmPredictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-ada-001", max_token=tokens)) #text-davinci-003

    #load data
    docs = SimpleDirectoryReader(LOCAL_PATH + 'data/source_of_knowledge').load_data() # make sure only the text files are in the folder

        
    vectorIndex = GPTVectorStoreIndex(documents=docs, llm_predictor=llmPredictor, prompt_helper=prompt_helper)
    vectorIndex.save_to_disk(vIdx)

    return vectorIndex

def qNa():
    if os.path.isfile(vIdx):
        vIndex = GPTSimpleVectorIndex.load_from_disk(vIdx)
        while True:
            prompt = input('Please ask your question here: ')
            if prompt.lower() != "goodbye.":
                response = vIndex.query(prompt, response_mode="compact")
                print(f"Response: {response} \n")
            else:
                print("Bot:- Goodbye!")
                break
    return "No source of knowledge found. Please upload documents first."

# vectorIndex = createVectorIndex('source_of_knowledge/data')
# qNa()

def qNa_source_of_knowledge(question):
    if os.path.isfile(vIdx):
        vIndex = GPTSimpleVectorIndex.load_from_disk(vIdx)
        response = vIndex.query(question, response_mode="compact")
        return response
    return "No source of knowledge found. Please upload documents first."

# # example
# question = "what do you know about onex" 
# print(qNa_source_of_knowledge(question))


def chatGPT3_response(user_input):
    openai.api_key = OpenAI_API_KEY_PERSONAL
    res = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": user_input}]
        )
    return res["choices"][0]["message"]["content"]

# # Example usage
# prompt = "What is the capital of France?"
# response = chatGPT3_response(prompt)
# print(response)


def create_embedding(name):
    chroma_client = chromadb.Client()
    return chroma_client.create_collection(name)

def query_pages(collection, urls, questions):
    docs = TrafilaturaWebReader().load_data(urls)
    index = GPTVectorStoreIndex.from_documents(docs, chroma_collection=collection)
    query_engine = index.as_query_engine()
    for question in questions:
        print(f"Question: {question} \n")
        print(f"Answer: {query_engine.query(question)}")


# # test case
# urls = ["https://mehrdad-zade.github.io/", "https://mehrdad-zade.github.io/#Experience"]    
# questions = ["tell me a bit about mehrdad's background what kind of job would he be suitable for"]
# collection = create_embedding("supertype")
# query_pages(
#         collection,
#         urls,
#         questions
#     )