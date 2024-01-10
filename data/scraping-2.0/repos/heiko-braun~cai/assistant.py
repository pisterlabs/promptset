from openai import OpenAI
from util.utils import show_json, as_json
import time
import jmespath
import json
import httpx

from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

from conf.constants import *

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from statemachine import State
from statemachine import StateMachine

import argparse
import cohere

import streamlit as st
from abc import ABC, abstractmethod

# ---

def get_response(client, thread):
    return client.beta.threads.messages.list(thread_id=thread.id, order="asc")

def pretty_print(messages):
    print("# Messages")
    for m in messages:
        print(f"{m.role}: {m.content[0].text.value}")
    print()

# primitive wait condition for API requests, needs improvement
def wait_on_run(client, run, thread):
    while run.status == "queued" or run.status == "in_progress":
        run = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id,
        )
        print("Thinking ... ", run.status)
        #show_json(run)
        time.sleep(0.5)        
    return run    

# fetch the call arguments from an assistant callback
def get_call_arguments(run):    
    #show_json(run)
    tool_calls = jmespath.search(
        "required_action.submit_tool_outputs.tool_calls", 
        as_json(run)
    )
    
    call_arguments = []
    for call in tool_calls:
        id = jmespath.search("id", call)    
        arguments = jmespath.search("function.arguments", call)        
        call_arguments.append(
            {
                "call_id": id,
                "call_arguments":json.loads(arguments)
            }
        )
    return call_arguments 
    
# search local storage for documentation related to componment    
def fetch_docs(entities):  
    print("Fetching docs for query: ", entities)  
    
    query_results = query_qdrant(entities, collection_name="camel_docs")
    num_matches = len(query_results)
    
    # print("First glance matches:")
    # for i, article in enumerate(query_results):    
    #    print(f'{i + 1}. {article.payload["filename"]} (Score: {round(article.score, 3)})')

    if num_matches > 0:

        docs = []
        for _, article in enumerate(query_results):
            with open(article.payload["filename"]) as f:                
                docs.append(f.read())

        # apply reranking
        co = cohere.Client(os.environ['COHERE_KEY'])
        rerank_hits = co.rerank(
            model = 'rerank-english-v2.0',
            query = entities,
            documents = docs,
            top_n = 3
        )

        print("Reranked matches: ")   
        for hit in rerank_hits:
            orig_result = query_results[hit.index]
            print(f'{orig_result.payload["filename"]} (Score: {round(hit.relevance_score, 3)})')            

        # TODO: This is wrong and needs to be fixed. it must consider the rerank
        doc = query_results[0]
        with open(doc.payload["filename"]) as f:
            contents = f.read()
            return contents
    else:
        return "No matching file found for "+entities  

def fetch_and_rerank(entities, collections, feedback):
    response_documents = []
    first_iteration_hits = []

    # lookup across multiple vector store
    for collection_name in collections:
            
        query_results = query_qdrant(entities, collection_name=collection_name)
        num_matches = len(query_results)
        
        
        feedback.print("First glance matches ("+collection_name+"):")
        for i, article in enumerate(query_results):    
           feedback.print(f'{i + 1}. {article.payload["metadata"]["page_number"]} (Score: {round(article.score, 3)})')
           first_iteration_hits.append(
               {
                   "content": article.payload["page_content"],
                   "ref": article.payload["metadata"]["page_number"]
               }
           )


    # apply reranking
    hit_contents = []
    for match in first_iteration_hits:                             
        hit_contents.append(match["content"])            
    
    co = cohere.Client(os.environ['COHERE_KEY'])
    rerank_hits = co.rerank(
        model = 'rerank-english-v2.0',
        query = entities,
        documents = hit_contents,
        top_n = 5
    )
    
    # the final results
    second_iteration_hits = []
    feedback.print("Reranked matches: ") 
    for i, hit in enumerate(rerank_hits):                
        orig_result = first_iteration_hits[hit.index]

        doc = Document(
            page_content= first_iteration_hits[hit.index]["content"], 
            metadata={
                "page_number": first_iteration_hits[hit.index]["ref"]
            }
        )

        second_iteration_hits.append(str(doc)) # TODO, inefficient but the StreamlitCallback Handler expects this text structure

        feedback.print(f'{orig_result["ref"]} (Score: {round(hit.relevance_score, 3)})')         

    # squash into single response 
    response_documents.append(' '.join(second_iteration_hits))

    return ' '.join(response_documents)
                
def query_qdrant(query, top_k=5, collection_name="fuse_camel_development"):
    openai_client = create_openai_client()
    qdrant_client = create_qdrant_client()

    embedded_query = get_embedding(openai_client=openai_client, text=query)
    
    query_results = qdrant_client.search(
        collection_name=collection_name,
        query_vector=(embedded_query),
        limit=top_k,
    )
    
    return query_results

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_embedding(openai_client, text, model="text-embedding-ada-002"):
   start = time.time()
   text = text.replace("\n", " ")
   resp = openai_client.embeddings.create(input = [text], model=model)
   print("Embedding ms: ", time.time() - start)
   return resp.data[0].embedding

def create_openai_client():
    client = OpenAI(
        timeout=httpx.Timeout(
            10.0, read=8.0, write=3.0, connect=3.0
            )
    )
    return client

def create_qdrant_client(): 
    client = QdrantClient(
       QDRANT_URL,
        api_key=QDRANT_KEY,
    )
    return client
            

# rewrite a question using the chat API
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(1))
def rewrite_question(openai_client, text):
        
    template = PromptTemplate.from_template(
        """
        Rephrase the following text:
        
        Text: \"\"\"{text}\"\"\"

        """
    )
       
    response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo-1106",
            messages=[
                {"role": "system", "content": "You are a service used to rewrite text"},
                {"role": "user", "content": template.format(text=text)}
            ]
        )
    
    return response.choices[0].message.content            
# ---

class StatusStrategy(ABC):
    @abstractmethod
    def print(self, message) -> str:
        pass

    @abstractmethod    
    def set_visible(self, is_visible):
        pass

    @abstractmethod    
    def set_tagline(self, tagline):
        pass
        
class LoggingStatus(StatusStrategy):
    def print(self, message):
        print(message)

    def set_visible(self, is_visible):
        pass  

    def set_tagline(self, tagline):
        pass  

class StreamlitStatus(StatusStrategy):
    
    def __init__(self, st_callback):
        self.st_callback = st_callback
        self.st_status = None
    
    def print(self, message):
        if(self.st_status is not None):
           self.st_status.write(message) 

    def set_visible(self, is_visible):
        if (is_visible):
            self.st_status = self.st_callback.status("Thinking ...") 
        else:            
            self.st_status.update(label="Completed!", state="complete", expanded=False)

    def set_tagline(self, tagline):
        if(self.st_status is not None):
           self.st_status.update(label=tagline) 

class Assistant(StateMachine):
    "Assistant state machine"
    
    feedback = LoggingStatus()
    status = None

    prompt = State(initial=True)
    running = State()
    lookup = State()
    answered = State(final=True)

    kickoff = prompt.to(running)
    request_docs = running.to(lookup)
    docs_supplied = lookup.to(running)
    resolved = running.to(answered)

    def __init__(self, st_callback=None):
        
        # streamlit callback, if present
        self.st_callback = st_callback

        if(self.st_callback is not None):
            self.feedback = StreamlitStatus(self.st_callback)

        # internal states
        self.prompt_text = None
        self.thread = None
        self.run = None
        
        self.openai_client = create_openai_client()
        self.lookups_total = 0        
        
        super().__init__()

    def display_message(self, message):
        if self.st_callback is not None:
            st.session_state.messages.append({"role": "assistant", "content": message})
            with st.chat_message("assistant"):
                st.markdown(message, unsafe_allow_html=True)  

    def display_status(self, message): 
        if(self.status is None):
            print(message)
        else:
            self.status.write('Lookup additional information ...')        
    
    def on_exit_prompt(self, text):
        self.lookups_total = 0
        self.prompt_text = text          

        # clear screen
        if(self.st_callback is not None):
            self.st_callback.empty()

        # display status widget
        self.feedback.set_visible(True)
            
        # start a new thread and delete old ones if exist
        if(self.thread is not None):
            self.openai_client.beta.threads.delete(self.thread.id)
        
        self.thread = self.openai_client.beta.threads.create()
        self.feedback.print("New Thread: " + str(self.thread.id)) 

        # Add initial message
        improved_question = rewrite_question(openai_client = self.openai_client, text=text)
        print("Improved question: \n", improved_question)
        message = self.openai_client.beta.threads.messages.create(
            thread_id=self.thread.id,
            role="user",
            content=improved_question,
        )

        # create a run
        self.run = self.openai_client.beta.threads.runs.create(
            thread_id=self.thread.id,
            assistant_id=ASSISTANT_ID,
        )        

    def on_enter_lookup(self):
        
        self.feedback.set_tagline("Working ...")
        self.feedback.print("Lookup additional information ...")
        
        self.lookups_total = self.lookups_total +1

        # take call arguments and invoke lookup
        args = get_call_arguments(self.run)
        outputs=[]              
        
        for a in args:
            entity_args = a["call_arguments"]["entities"]
            self.feedback.print("Keywords: " + ' | '.join(entity_args)       )
 
            keywords = ' '.join(entity_args)
            # it often includes camel itself. remove it
            keywords = keywords.replace('Apache', '').replace('Camel', '')

            # we may end up with no keywrods at all
            if(len(keywords)==0 or keywords.isspace()):
                outputs.append(
                    {
                        "tool_call_id": a["call_id"],
                        "output": "'No additional information found.'"
                    }
                )
                continue

            
            #doc = fetch_pdf_pages(entities=keywords, feedback=self.feedback)
            doc = fetch_and_rerank(
                entities=keywords, 
                collections=["rhaetor.github.io_2", "rhaetor.github.io_components_2"],
                feedback=self.feedback
                )
            outputs.append(
                {
                    "tool_call_id": a["call_id"],
                    "output": "'"+doc+"'"
                }
            )
        
        # submit lookup results (aka tool outputs)
        self.run = self.openai_client.beta.threads.runs.submit_tool_outputs(
            thread_id=self.thread.id,
            run_id=self.run.id,
            tool_outputs=outputs
            )    
        
        self.docs_supplied()
        self.feedback.print("Processing new information ...")

    # starting a thinking loop    
    def on_enter_running(self):

        self.feedback.set_tagline("Thinking ...")    

        self.run = wait_on_run(self.openai_client, self.run, self.thread)        

        if(self.run.status == "requires_action"):            
            self.request_docs()
        elif(self.run.status == "completed"):    
            self.resolved()
        else:
            self.feedback.print("Illegal state: " + self.run.status)            
            print(self.run.last_error)
            if (self.st_callback is not None):
                self.st_callback.error(self.run.last_error)
            
    # the assistant has resolved the question
    def on_enter_answered(self):

        # thread complete, show answer
        assistant_response = get_response(self.openai_client, self.thread)
        for m in assistant_response:
            if(m.role == "assistant"):
                self.display_message(m.content[0].text.value)    

        pretty_print(assistant_response)

        # delete the thread
        self.openai_client.beta.threads.delete(self.thread.id)        
        self.feedback.print("Deleted Thread: " + str(self.thread.id))
        self.feedback.set_visible(False)
        self.thread = None

# --

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Camel Support Assistant')
    parser.add_argument('-f', '--filename', help='The input file that will be taken as a prompt', required=False)
    args = parser.parse_args()

    if(args.filename == None):
        prompt = input("Prompt: ")  
    else:
        with open(args.filename) as f:
            prompt = f.read()        
        prompt = prompt.replace('\n', ' ').replace('\r', '')        
        
        
    sm = Assistant()        
    sm.kickoff(prompt)

