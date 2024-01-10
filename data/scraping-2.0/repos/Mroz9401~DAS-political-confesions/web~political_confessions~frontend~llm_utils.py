import openai
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationSummaryMemory, ConversationSummaryBufferMemory, ConversationKGMemory
import os
# import torch
from transformers import AutoTokenizer, pipeline, AutoModel
import math
import json
import numpy as np
from langchain.prompts.prompt import PromptTemplate
from .pipeline import EmbeddingProcessor, PromptProcessor, open_pkl


def open_file(filepath):
    with open(filepath, 'r') as infile:
        return infile.read()
    
def get_llm_response(input_message, key_path, embedding_path, model_name, template_prompt):
    key_content = open_file(key_path)
    OPENAI_API_KEY = str(key_content.strip())
    embeddings = open_pkl(embedding_path)
    
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    feature_extraction_pipeline = pipeline('feature-extraction', model=model, tokenizer=tokenizer)

    llm = ChatOpenAI(temperature=0.77, model_name="ft:gpt-3.5-turbo-1106:personal::8V1zpYfy", openai_api_key = OPENAI_API_KEY)
    memory_llm = ChatOpenAI(temperature=0.9, model_name='gpt-3.5-turbo-1106', openai_api_key = OPENAI_API_KEY)
    #template of a prompt
    # template = template_prompt + ": " + input_message + "Shrnutí konverzace: " + "Petr Fiala:"
    template = """Jsi premiér České republiky Petr Fiala. Máš dán kontext, odpověz jako Petr Fiala na následující prompt: {input}.
    Shrnutí konverzace:
    {history}
    Petr Fiala:"""

    context=[]

    PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
    conversation = ConversationChain(
        prompt=PROMPT,
        llm=llm,
        verbose=False,
        memory=ConversationSummaryMemory(llm=memory_llm, ai_prexis= "Petr Fiala"),
    )

    prompt_processer = PromptProcessor(pipeline=feature_extraction_pipeline)
    formatted_prompt = prompt_processer.format_prompt(input_message, embeddings, threshold = 0.3)
    print(formatted_prompt)
    response = conversation.predict(input=formatted_prompt)

    return response

