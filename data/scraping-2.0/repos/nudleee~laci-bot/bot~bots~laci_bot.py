from botbuilder.core import ActivityHandler, TurnContext, ConversationState
from botbuilder.schema import ChannelAccount
import os 
import openai
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, CSVLoader, TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import  ConversationalRetrievalChain, LLMChain, StuffDocumentsChain

from data_models import ConversationHistory

openai.api_key  = os.environ['OPENAI_API_KEY']

if os.name == 'posix':
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

class LaciBot(ActivityHandler):
    def __init__(self, conversation_state: ConversationState):
        if conversation_state is None:
              raise TypeError(
                    "[LaciBot]: Missing parameter. conversation_state is required but None was given"
              )         
        self._conversation_state = conversation_state
        self.conversation_history_accessor = self._conversation_state.create_property("ConversationHistory")
        self.persist_directory = 'files/chroma'
        self.embedding = OpenAIEmbeddings()
        self.vector_db = self.load_db()
        self.chain = ConversationalRetrievalChain(
                combine_docs_chain=self.doc_chain,
                question_generator=self.question_generator_chain,
                retriever=self.vector_db.as_retriever(search_type="mmr", search_kwargs={"fetch_k":8, "k": 5}),
            )        
        

    llm_name = 'gpt-3.5-turbo'
    llm = ChatOpenAI(model=llm_name, temperature=0.3, request_timeout=120)   

    def load_db(self):
        if os.path.exists(self.persist_directory):
            return Chroma(persist_directory=self.persist_directory,embedding_function=self.embedding)
        else:
            loader_kwargs={"encoding": "utf_8"}
            pdf_loader = PyPDFLoader('files/Szakmai_gyak_BSc_szabályzat_2014 után.pdf')
            pdf_docs = pdf_loader.load()
            csv_loader = DirectoryLoader('files/', glob="*.csv", loader_cls=CSVLoader, loader_kwargs=loader_kwargs)
            csv_docs = csv_loader.load()
            txt_loader = DirectoryLoader('files/', glob='*.txt', loader_cls=TextLoader, loader_kwargs=loader_kwargs)
            txt_docs=txt_loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = 800,
                chunk_overlap = 100,
                length_function=len,
                is_separator_regex=True,
                separators=["\n\s*\n", "\n\s*", "\n", "\n\n\n\n", "\n\n\t\t"]
            )
            pdf_splits = text_splitter.split_documents(pdf_docs)
            txt_splits = text_splitter.split_documents(txt_docs)
            data = []
            data.extend(pdf_splits)
            data.extend(txt_splits)
            data.extend(csv_docs)
            vector_db = Chroma.from_documents(documents=data, embedding=self.embedding, persist_directory=self.persist_directory)
            vector_db.persist()
            return vector_db
        
   

    template = """A BME VIK szakmai gyakorlattal kapcsolatos kérdéseket megválaszoló chatbot vagy. A feladatod, hogy a kérdést alakítsd át az előzmények alapján, hogy értelmes legyen. 
    Ha nem tudsz a szakmai gyakorlatra vonatkozó kérdést előállítani, akkor legyen az új kérdés: "Nem tudok válaszolni".

    Előzmények: {chat_history}

    Kérdés: {question}

    Új kérdés:"""

    QG_CHAIN_PROMPT = PromptTemplate(input_variables=['chat_history', 'question'],template=template)
    question_generator_chain = LLMChain(llm=llm, prompt=QG_CHAIN_PROMPT)

    qa_template = """A BME VIK szakmai gyakorlattal kapcsolatos kérdéseket megválaszoló chatbot vagy és a neved Laci-bot. Válaszolj a kérdésre magyarul, amihez az alábbiakban találasz releváns információkat,
    de ha nem tudsz válaszolni, akkor ne próbálj meg kitalálni valamit, hanem mondjad "Sajnos nem tudok ezzel kapcsolatban információval szolgálni".

    Dokumentumok: {documents}

    Előzmények: {chat_history}

    Kérdés: {question}

    Válasz:"""

    QA_CHAIN_PROMPT = PromptTemplate(input_variables=['context', 'chat_history', 'documents'],template=qa_template)
    llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT)
    doc_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name='documents',)
    
    async def on_turn(self, turn_context: TurnContext):
            await super().on_turn(turn_context)
            await self._conversation_state.save_changes(turn_context)

    async def on_message_activity(self, turn_context: TurnContext):
        conversation_history = await self.conversation_history_accessor.get(turn_context, ConversationHistory)   

        if conversation_history._memory is None:        
            conversation_history._memory=ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        
        message = turn_context.activity.text
        self.chain.memory = conversation_history._memory
        chat_history = self.chain.memory.load_memory_variables({})['chat_history']
        result = self.chain({'question': message, 'chat_history': chat_history})
        await turn_context.send_activity(result['answer'])

                 

        

        
    

    