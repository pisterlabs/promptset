from dataclasses import dataclass

import requests
import json

from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.document_loaders import WebBaseLoader
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory, ConversationSummaryBufferMemory, \
    ChatMessageHistory
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate
from langchain.schema import messages_to_dict, messages_from_dict
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.vectorstores import Chroma
from langchain.embeddings import OllamaEmbeddings
from langchain import hub
from langchain.llms import Ollama
from pydantic import BaseModel

from llm_utils import LlmEmbedding
from sqlite_lit import SqliteMemDbContext

url = "http://localhost:11434/api/generate"

headers = {
    'Content-Type': 'application/json',
}

conversation_history = []


@dataclass
class Conversation:
    role_name: str
    content: str


class ChatHistoryAgentBase:
    def load_chat_history_data(self, conversation_id: int) -> list[Conversation]:
        pass

    def append_chat_history(self, conversation_id: int, item: Conversation):
        pass


class ChatHistoryMemoryAgent(ChatHistoryAgentBase):
    def __init__(self):
        self.db = SqliteMemDbContext()
        self.db.execute("""
        CREATE TABLE IF NOT EXISTS ConversationMessages (
            id INTEGER PRIMARY KEY,
            conversation_id INTEGER,
            role_name TEXT,
            content TEXT
        )
        """)

    def append_chat_history(self, conversation_id: int, item: Conversation):
        db = self.db
        db.execute("INSERT INTO ConversationMessages(conversation_id, role_name, content) VALUES(?, ?, ?)",
                   (conversation_id, item.role_name, item.content))

    def load_chat_history_data(self, conversation_id: int) -> list[Conversation]:
        db = self.db
        result = db.query_objects("SELECT role_name, content FROM ConversationMessages WHERE conversation_id=? ORDER BY id DESC LIMIT 10",
                                  (conversation_id, ))
        result = result[::-1]
        return result

    def extract_chat_history(self, chain):
        extracted_messages = chain.memory.chat_memory.messages
        # [HumanMessage(content='what do you know about Python in less than 10 words', additional_kwargs={}),
        #  AIMessage(content='Python is a high-level programming language.', additional_kwargs={})]
        ingest_data = messages_to_dict(extracted_messages)
        # [{'type': 'human',
        #   'data': {'content': 'what do you know about Python in less than 10 words',
        #    'additional_kwargs': {}}},
        #  {'type': 'ai',
        #   'data': {'content': 'Python is a high-level programming language.',
        #    'additional_kwargs': {}}}]
        items = []
        for row in ingest_data:
            item = Conversation(
                role_name=row['type'],
                content=row['data']['content']
            )
            items.append(item)
        return items

    def load_conversation_buffer_memory(self, conversation_id: int) -> ConversationBufferMemory:
        rows = self.load_chat_history_data(conversation_id)
        messages = []
        for item in rows:
            messages.append({
                'type': item.role_name,
                'data': {
                    'content': item.content,
                    'additional_kwargs': {}
                }
            })
        retrieved_messages = messages_from_dict(messages)
        retrieved_chat_history = ChatMessageHistory(messages=retrieved_messages)
        retrieved_memory = ConversationBufferMemory(chat_memory=retrieved_chat_history,
                                                    memory_key="history",
                                                    input_key="question")
        return retrieved_memory


def generate_response(prompt):
    conversation_history.append(prompt)

    full_prompt = "\n".join(conversation_history)

    data = {
        "model": "mistral",
        "stream": False,
        "prompt": full_prompt,
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        response_text = response.text
        data = json.loads(response_text)
        actual_response = data["response"]
        conversation_history.append(actual_response)
        return actual_response
    else:
        print("Error:", response.status_code, response.text)
        return None

def chat():
   while True:
      user_input = input("請輸入字串: ")
      if user_input == "/bye":
         break
      resp = generate_response(user_input)
      print(resp)
   

def qa_docs():
   loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
   data = loader.load()
   text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50)
   all_splits = text_splitter.split_documents(data)
   
   llm_embedding = LlmEmbedding("../models/BAA_Ibge-large-en-v1.5")
   vectorstore = Chroma.from_documents(documents=all_splits,
                                       embedding=llm_embedding.embedding)

   ## Retrieve
   # question = "How can Task Decomposition be done?"
   # docs = vectorstore.similarity_search(question)                                       
   QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-llama")

   ollama = Ollama(base_url='http://localhost:11434', model="mistral")
   qachain = RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())
   while True:
      user_input = input("請輸入字串: ")
      if user_input == "/bye":
         break
      resp = qachain({"query": user_input})
      print(resp['result'])



def load_vector_store():
    loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50)
    all_splits = text_splitter.split_documents(data)
    llm_embedding = LlmEmbedding("../models/BAA_Ibge-large-en-v1.5")
    vectorstore = Chroma.from_documents(documents=all_splits,
                                        embedding=llm_embedding.embedding)
    return vectorstore

def qa_mem1():
    # not work
    vectorstore = load_vector_store()

    _template = """Given the following conversation and a follow up question, 
    rephrase the follow up question to be a standalone question.
    Chat History:
    {chat_history}
    Follow Up Input: {question}
    Standalone question:"""
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

    ollama = Ollama(base_url='http://localhost:11434', model="mistral")

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True, output_key='answer')

    chain = ConversationalRetrievalChain.from_llm(
        ollama,
        vectorstore.as_retriever(),  # see below for vectorstore definition
        memory=memory,
        condense_question_prompt=CONDENSE_QUESTION_PROMPT,
        get_chat_history=lambda h: h,
    )

    while True:
        print("")
        user_input = input("qa_mem: ")
        if user_input == "/bye":
            break
        resp = chain({
            "question": user_input,
            "chat_history": [
               ("What is the date Australia was founded?", "Australia was founded in 1901."),
            ]},
            return_only_outputs=True)
        print(f"{chain.memory.chat_memory.messages=}")
        print(resp['answer'])



def qa_mem2():
    # not worked
    vectorstore = load_vector_store()
    ollama = Ollama(base_url='http://localhost:11434', model="mistral")
    memory = ConversationSummaryBufferMemory(
        llm=ollama,
        output_key='answer',
        memory_key='chat_history',
        return_messages=True)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3, "include_metadata": True})

    chain = ConversationalRetrievalChain.from_llm(
        ollama,
        retriever,
        get_chat_history=lambda h: h,
        memory=memory)

    chat_history = []
    while True:
        user_input = input("mem2: ")
        if user_input == "/bye":
            break
        resp = chain({"question": user_input, "chat_history": chat_history})
        answer = resp['answer']
        chat_history.append((user_input, answer))
        print(answer)

def qa_mem3():
    vectorstore = load_vector_store()

    template = """
    Use the following context (delimited by <ctx></ctx>) and the chat history (delimited by <hs></hs>) to answer the question:
    ------
    <ctx>
    {context}
    </ctx>
    ------
    <hs>
    {history}
    </hs>
    ------
    {question}
    Answer:
    """
    prompt = PromptTemplate(
        input_variables=["history", "context", "question"],
        template=template,
    )

    ollama = Ollama(base_url='http://localhost:11434', model="mistral:instruct")

    chat_history_memory_agent = ChatHistoryMemoryAgent()
    conversation_id = 123
    chat_history_memory_agent.append_chat_history(conversation_id, Conversation(
        role_name='human',
        content='Who is Flash?'
    ))
    chat_history_memory_agent.append_chat_history(conversation_id, Conversation(
        role_name='ai',
        content='He is superman.'
    ))
    memory = chat_history_memory_agent.load_conversation_buffer_memory(conversation_id)

    chain = RetrievalQA.from_chain_type(
        llm=ollama,
        chain_type='stuff',
        retriever=vectorstore.as_retriever(),
        verbose=False,
        chain_type_kwargs={
            # "verbose": True,
            "prompt": prompt,
            "memory": memory,
        }
    )

    history = []
    history.append({'Human: Who is Flash?'})
    history.append({'AI: Flash is superman. '})
    while True:
        print("")
        user_input = input("qa_mem: ")
        if user_input == "/bye":
            break
        chat_history_memory_agent.append_chat_history(conversation_id, Conversation(
            role_name='human',
            content=user_input
        ))
        resp = chain({
                "query": user_input,
                'history': history
            })
        answer = resp['result']
        print(answer)
        chat_history_memory_agent.append_chat_history(conversation_id, Conversation(
            role_name='ai',
            content=answer
        ))

#qa_docs()
#qa_mem1()
qa_mem3()
