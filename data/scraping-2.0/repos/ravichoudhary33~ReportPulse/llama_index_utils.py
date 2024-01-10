import os
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from prompts import SUMMARY_PROMPT
import redis
import hashlib
from llama_index import StorageContext, load_index_from_storage
import openai

openai.api_key = os.environ["OPENAI_API_KEY"]
# Connect to Redis
r = redis.Redis(host='localhost', port=6379, db=0)

def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature, # this is the degree of randomness of the model's output
    )
#     print(str(response.choices[0].message))
    return response.choices[0].message["content"]

class ReportPulseAssistent:

    def __init__(self, data_dir,lang="ENGLISH", use_openai=True) -> None:
        # on initialization
        self.lang = lang
        self.doc_text = None
        self.documents = self.get_docs(data_dir)
        self.system_prompt = f"""
            You are an ai health assistant that understand medical report and speaks in laymen term.'
            Given the medical report of a person delimited by ```, \
            Generate a reply to expalin the medical report in laymen terms for their report.\
            Make sure to use specific details from the report.
            Write in a concise and professional tone.
            Patient report: ```{self.doc_text}```
            """
        self.msgContext =  [  
            {'role':'system', 'content': self.system_prompt},    
            {'role':'user', 'content':f'summarise the report in laymen term with {lang} language.'}]
        
        if not use_openai:
            self.index = self.get_index(self.documents)
            self.chat_engine = self.index.as_chat_engine(verbose=True)


    def get_docs(self, data_dir):
        # data_dir = '../data'
        documents = SimpleDirectoryReader(data_dir).load_data()
        docs = []
        for doc in documents:
            docs.append(doc.text)
        doc_text = ' '.join(docs).replace('\n', ' ').strip()
        self.doc_text = doc_text
        return documents


    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(3))
    def get_index(self, documents):

        # rebuild storage context
        try:
            storage_context = StorageContext.from_defaults(persist_dir='./storage')
            index = load_index_from_storage(storage_context)
        except Exception as e:
            index = GPTVectorStoreIndex.from_documents(documents)
            index.storage_context.persist()
        
        return index

    def get_next_message(self, prompt=SUMMARY_PROMPT, lang="ENGLISH", prompt_type='summary', use_openai=True):

        prompt_enc = prompt + self.lang 
        md5_hash = hashlib.md5(prompt_enc.encode()).hexdigest()

        if r.exists(md5_hash):
            response = r.get(md5_hash).decode('utf-8')
            return response
        else:
            if use_openai:
                if prompt_type == 'summary':
                    response = get_completion_from_messages(self.msgContext, temperature=0)
                    self.msgContext.extend([
                        {'role':'assistant', 'content': response}    
                    ])
                    r_res = response.encode('utf-8')
                    r.set(md5_hash, r_res)
                    return response
                elif prompt_type == 'report':
                    user_prompt = f"""
                    Extract the user report as list of json object with keys as "Parameter", "Result", "Biological Ref Range".
                    Sample Output:\
                    [{{
                        "Parameter": "Absolute Eosinophils",
                        "Result": "213.6",
                        "Biological Ref Range": "20-500 /cmm"
                    }}]
                    """
                    if self.lang != "ENGLISH":
                        user_prompt += f"Translate the output in {self.lang}. And Output only valid json."
                    self.msgContext.extend([
                        {'role':'user', 'content':user_prompt}     
                    ])
                    response = get_completion_from_messages(self.msgContext, temperature=0)
                    self.msgContext.extend([
                        {'role':'assistant', 'content': response}    
                    ])
                    r_res = response.encode('utf-8')
                    r.set(md5_hash, r_res)
                    return response
                else:
                    if self.lang != "ENGLISH":
                        prompt += f"Translate the output in {self.lang}."
                    self.msgContext.extend([
                        {'role':'user', 'content':prompt} 
                    ])
                    response = get_completion_from_messages(self.msgContext, temperature=0)
                    self.msgContext.extend([
                        {'role':'assistant', 'content': response}    
                    ])
                    r_res = response.encode('utf-8')
                    r.set(md5_hash, r_res)
                    return response


