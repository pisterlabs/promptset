'''
pip install gpt_index
pip install langchain
pip install transformers
'''

from gpt_index import SimpleDirectoryReader, GPTListIndex, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
import sys
import os
from secrets_custom import credentialsOpenAI

def createVectorIndex(path):

    os.environ["OPENAI_API_KEY"] = credentialsOpenAI

    max_input = 4096
    tokens = 256
    chunk_size = 600
    max_chunk_overlap = 20

    prompt_helper = PromptHelper(max_input, tokens, max_chunk_overlap, chunk_size_limit=chunk_size)

    #define LLM
    llmPredictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-ada-001", max_token=tokens)) #text-davinci-003

    #load data
    docs = SimpleDirectoryReader(path).load_data()

        
    vectorIndex = GPTSimpleVectorIndex(documents=docs, llm_predictor=llmPredictor, prompt_helper=prompt_helper)
    vectorIndex.save_to_disk('source_of_knowledge/vectorIndex.json')

    return vectorIndex

def qNa(vectorIndex):
    vIndex = GPTSimpleVectorIndex.load_from_disk(vectorIndex)
    while True:
        prompt = input('Please ask your question here: ')
        if prompt.lower() != "goodbye.":
            response = vIndex.query(prompt, response_mode="compact")
            print(f"Response: {response} \n")
        else:
            print("Bot:- Goodbye!")
            break

# vectorIndex = createVectorIndex('source_of_knowledge/data')
# qNa('source_of_knowledge/vectorIndex.json')