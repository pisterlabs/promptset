from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from concurrent.futures import ThreadPoolExecutor
from PyPDF2 import PdfReader
import io, asyncio
from wandb.integration.langchain import WandbTracer
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI, ChatAnthropic, AzureChatOpenAI
from langchain.llms import Bedrock
import boto3
import os
import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA, LLMSummarizationCheckerChain
from summary_prompts import get_guidelines
import logging
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.docstore.document import Document
from utils import split_text
from botocore.exceptions import ClientError
from botocore.config import Config
import logging
import json

import PyPDF2

def create_bedrock_connection():
        config=Config(connect_timeout=10, read_timeout=300, retries={'max_attempts': 5})
        return boto3.client("bedrock-runtime", "us-west-2", config=config)
 
def create_claude_summary( client_bedrock, prompt):
      model="anthropic.claude-v2"
      max_tokens_to_sample=50000
      temperature=0.3
      top_k=250
      top_p=0.999
      stop_sequences=["\n\nHuman:"]
      try:
          body = json.dumps(
              {
                  "prompt": f"\n\nHuman:{prompt}\n\nAssistant:",
                  "max_tokens_to_sample": max_tokens_to_sample,
                  "temperature": temperature,
                  "top_k": top_k,
                  "top_p": top_p,
                  "stop_sequences": stop_sequences,
              }
          )
          logging.debug(f"Request body: {body}")

          accept = "application/json"
          content_type = "application/json"

          response = client_bedrock.invoke_model(
              body=body, modelId=model, accept=accept, contentType=content_type
          )
          logging.debug(f"{model} called...")
          response_body = json.loads(response.get("body").read())
          logging.info(f"Response body: {response_body}")

          # remove the first line of text that explains the task completed
          # e.g. " Here are three hypothetical questions that the passage could help answer:\n\n"
          formatted_response = (
              response_body.get("completion").split("\n", 2)[2].strip()
          )
          return formatted_response
      except ClientError as ex:
          logging.error(ex)
          exit(1)

async def summarise_documents(texts=None, docs=None):
  # Initialize a ThreadPoolExecutor
    if texts is not None:
       docs = split_text(texts)
    pdfReader = None
    executor = ThreadPoolExecutor(max_workers=10)
    # creating a pdf file object
    

    # Split the PDF document into pages
    pdf_pages = []

  
    
    #for doc in docs:
    #    pdf_pages.append(doc.page_content)
    llm = return_llm()    
    chain = load_summarize_chain(llm, chain_type="stuff")
    answer = chain.run(docs)
        

#
    ## Call OpenAI to summarize each page in parallel
    #summary_futures = [executor.submit(call_llm_summarize, page) for page in docs]
#
    ## Wait for all the summarisable_text
    #summarisable_text_coroutine = [future.result() for future in summary_futures]
#
    ## Use asyncio.gather to wait for all coroutines to complete
    #summarisable_text = await asyncio.gather(*summarisable_text_coroutine)
    ## Create LangChain documents from the summarisable_text
    #docs = [Document(summary) for summary in summarisable_text]
#
    ## Combine the summarisable_text as context into a new call
    #context = ' '.join(summarisable_text)
    #llm = return_llm()create_gpt_response
    #chain = load_summarize_chain(llm, chain_type="stuff")
    #answer = chain.run(docs)
    #answer = get_final_summary(docs)


    return answer

def create_gpt_response(prompt):
    openai.api_key = os.environ["OPENAI_API_KEY"]
    from openai import OpenAI
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {
            "role": "system",
            "content": "You are a helpful assistant "
            },
            {
            "role": "user",
            "content": prompt
            }
        ],
        temperature=0,
        max_tokens=4096,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
        )
    response_content = response.choices[0].message.content
    return response_content


def summarise_with_gpt_turbo(raw_text=None, texts=None, docs=None):
    if raw_text is not None:
        summarisable_text = raw_text
    elif texts is not None:
        summarisable_text = " ".join(texts)
    else:
        summarisable_text = ' '.join([doc.page_content for doc in docs])
    
    summary = ""

    summarisable_text = f"""
        ---
        summarisable_text
        {summarisable_text.strip()}
        ---
        """



    prompt = get_guidelines()["cod_summary_bullets"]
    prompt = prompt.format(context=summarisable_text)


    try:
        summary = create_gpt_response(prompt)
    except Exception as ex:
        logging.error(ex)
        summary = str(ex)
  

    

    return summary
def summarise_with_claude(raw_text=None, texts=None, docs=None):
    summarisable_text = ""
    if raw_text is not None:
        summarisable_text = raw_text
    elif texts is not None:
       summarisable_text = " ".join(texts)
    else:
       summarisable_text = ' '.join([doc.page_content for doc in docs])
       
    summary = ""
    
    client_bedrock = create_bedrock_connection()

    summarisable_text = f"""
        <summarisable_text>
        {summarisable_text.strip()}
        </summarisable_text>"""



    # Open the file in read mode and read its contents into a string


  
    #prompt = f"""Write a concise single-paragraph summary of the main points, events, and ideas covered in following individual chapter summarisable_text. 
    #    Construct a complete, grammatically-correct paragraph. DO NOT use bullet points. 
#
    #    <summarisable_text>
    #    {summarisable_text.strip()}
    #    </summarisable_text>"""
    prompt = get_guidelines()["cod_summary_bullets"]
    prompt = prompt.format(context=summarisable_text)
   
    try:
        summary = create_claude_summary(client_bedrock, prompt)
    except Exception as ex:
        logging.error(ex)
        summary = str(ex)
  

   
    return summary
    




def return_bedrock_llm():
  bedrockruntime = boto3.client(service_name='bedrock-runtime')
  llm = Bedrock(
      credentials_profile_name="default",
      model_id="anthropic.claude-v2",
      client=bedrockruntime
  )
  return llm

def return_llm(provider="openai"):
  if provider == "openai":
    if os.getenv('DEV_MODE'):
      wandb_config = {"project": "DiscordChatBot"}
      return ChatOpenAI(model_name="gpt-3.5-turbo-16k",callbacks=[WandbTracer(wandb_config)], temperature=0)
    else:
      return ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0)

def ask_claude(query):
    
    anthropic = Anthropic()
    completion = anthropic.completions.create(
    model="claude-2",
    max_tokens_to_sample=100000,
    prompt=f"{HUMAN_PROMPT} {query} {AI_PROMPT}",
    )
    return completion.completion





def get_final_summary(docs):

 
    if os.getenv('DEV_MODE'):
        wandb_config = {"project": "DiscordChatBot"}
        qa = RetrievalQA.from_chain_type(llm=return_llm(), chain_type="stuff",callbacks=[WandbTracer(wandb_config)])
        #LLMSummarizationCheckerChain.from_llm(llm, verbose=True, max_checks=2)
    else:
        qa = RetrievalQA.from_chain_type(llm=return_llm(), chain_type="stuff")
    


    query = get_guidelines()["cod_summary_bullets"]
    reduce_template = """The following is set of summarisable_text:
    {doc_summarisable_text}
    Take these and distill it into a final, consolidated summary of the main themes. 
    Helpful Answer:"""
    reduce_prompt = PromptTemplate.from_template(reduce_template)


    # Define LLM chain
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
    llm_chain = LLMChain(llm=llm, prompt=query)

    # Define StuffDocumentsChain
    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain, document_variable_name="doc_summarisable_text"
    )


        #query = "You are a helpful assistant with concise and accurate responses given in the tone of a professional presentation. Give and detailed Summary of this document making sure to include the following sections Title: Authors: Ideas: Conclusions:"
   # documents = retriever.get_relevant_documents(query)
    stuff_chain.run(docs)
    answer=qa.run(query)
    return answer
   

def retrieve_answer(vectorstore):
  llm = return_llm()  
  
  
    
  if os.getenv('DEV_MODE'):
    wandb_config = {"project": "DiscordChatBot"}
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":5}),callbacks=[WandbTracer(wandb_config)])
   #LLMSummarizationCheckerChain.from_llm(llm, verbose=True, max_checks=2)
  else:
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":5}))
  retriever=vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":20})
  
  guidelines = get_guidelines()["cod_summary_bullets"]
  query = guidelines
  #query = "You are a helpful assistant with concise and accurate responses given in the tone of a professional presentation. Give and detailed Summary of this document making sure to include the following sections Title: Authors: Ideas: Conclusions:"
  documents = retriever.get_relevant_documents(query)

  answer=qa.run(query)
  
  logging.info(answer)
  return answer