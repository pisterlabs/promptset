# Using a FIASS document store 
import os
from haystack.document_stores import FAISSDocumentStore

# Load the saved index into a new DocumentStore instance:
# Also, provide `config_path` parameter if you set it when calling the `save()` method: 
document_store = FAISSDocumentStore.load(index_path="docstore/my_index.faiss", config_path="docstore/my_config.json")

# Check if the DocumentStore is loaded correctly
assert document_store.faiss_index_factory_str == "Flat"


# Initilazing prompt node from the start to avoid delays later : 
from haystack.nodes import PromptNode,PromptTemplate
from haystack.pipelines import Pipeline

# # Initializing agent and tools 
# from haystack.agents import Agent, Tool
# from haystack.agents.base import ToolsManager


from haystack.nodes import EmbeddingRetriever
retriever = EmbeddingRetriever(
    document_store=document_store, embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1"
)
# This embedding retreiver gave the wrong file for "What are the different billing methdos in SD ??"

# from haystack.nodes import BM25Retriever
# retriever = BM25Retriever(document_store=document_store, top_k=2)


# qa_template = PromptTemplate(
#     name="Question_and_Answer",
#     prompt_text="""
#     You are an AI assistant. Your task is to use the content to give a detailed and easily understandable answer  
#     Content: {input}\n\n
#     Answer:
#     """
# )

lfqa_prompt = PromptTemplate(
    name="lfqa",
    prompt_text="""Synthesize a comprehensive answer from the following text for the given question. 
                             Provide a clear and concise response that summarizes the key points and information presented in the text. 
                             Your answer should be in your own words and be no longer than 50 words. 
                             \n\n Related text: {join(documents)} \n\n Question: {query} \n\n 
                             Final Answer:""",
)
api_key=os.environ.get("API-KEY")
prompt_node_working = PromptNode("gpt-3.5-turbo", api_key=api_key, default_prompt_template=lfqa_prompt,model_kwargs={"stream":True})
# prompt_node_working = PromptNode("openai-gpt", default_prompt_template=lfqa_prompt,model_kwargs={"stream":True})

# prompt_node = PromptNode("distilbert-base-cased-distilled-squad",default_prompt_template=lfqa_prompt,model_kwargs={"stream":True})

# from haystack.nodes import OpenAIAnswerGenerator
# generator = OpenAIAnswerGenerator(api_key="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")

# from haystack.pipelines import GenerativeQAPipeline

# pipeline = GenerativeQAPipeline(generator=generator, retriever=retriever)
# result = pipeline.run(query='How to create a sales order', params={"Retriever": {"top_k": 1}})

query_pipeline = Pipeline()
query_pipeline.add_node(component=retriever, name="Retriever", inputs=["Query"])

query_pipeline.add_node(component=prompt_node_working, name="prompt_node", inputs=["Retriever"])


## This works perfectly for lfqa, Maybe 


# Creating a funciton to integrate all this : 

def question_answering_bot(input_question):
    answer = query_pipeline.run(query=input_question, params={"Retriever": {"top_k": 3}})
    
    # # Assuming 'answer' is a Document object
    # response = {
    #     'text': answer.text,
    #     'start': answer.start,
    #     'end': answer.end,
    #     'score': answer.score,
    # }
    # return response
    
    return answer["results"]

    

# # Extract the 'content' value from each document
# contents = [doc.content for doc in result['documents']]

# # Print the contents
# for content in contents:
#     print(content)

# # Extract the 'content' value from each document
# contents = [doc.content for doc in result['documents']]

# # Join all the content values into a single string
# joined_content = '\n'.join(contents)

# result = prompt_node.prompt(prompt_template=qa_template, input=joined_content)
# print(result)
# query_pipeline.add_node(component=prompt_node, name="prompt_node", inputs=["Retriever"])

# hotpot_questions = [
#     "What are the different billing methods  ?"
# ]


# for question in hotpot_questions:
#     output = query_pipeline.run(query=question)
#     print(output["results"])


    # while(True):
    #     input_question = input("Enter the quesiton which you want to ask the bot : ")
    #     if(input_question=="#"):
    #         print("thank you ")
    #         break
    #     else:
    #         question_answering_bot(input_question)
    
    ## Testing llms 


# reply = question_answering_bot("Expalin the policies of performance security bond ?")
# print("\n\n\n RESULT\n")
# print(reply)
