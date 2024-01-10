import os
import pickle
from pathlib import Path
from operator import itemgetter

from langchain.chat_models import ChatOpenAI
from langchain.schema import format_document
from langchain.vectorstores.faiss import FAISS
from langchain.prompts import PromptTemplate, ChatPromptTemplate

from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.retrievers.multi_query import MultiQueryRetriever

from langchain_core.messages import get_buffer_string
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


class ChainPipe():
    
    def __init__(self, keyword):
        self.BASE = Path(__file__).parent.parent.parent / 'data' / 'templates' / 'chatgpt' 
        self.history_path = self.BASE / 'history' / keyword / f'{keyword}.pkl'
        self.database_path = self.BASE / 'database' / f'{keyword}'
        self.memory = None
        
    def load_history(self):
        if os.path.isfile(self.history_path):
            with open(self.history_path,'rb') as f:      #경로예시 : ./data/history/m.txt
                memory = pickle.load(f)
        else:
            memory = ConversationBufferMemory(
                return_messages=True, output_key="answer", input_key="question"
            )
        self.memory = memory
        return memory

    def make_chain(self):
        #stream_it = AsyncIteratorCallbackHandler()
        self.DOCUMENT_PROMPT = open(self.BASE / 'prom2.txt','r', encoding='UTF8').read()
        self.ANSWER_PROMPT = open(self.BASE / 'answer_prompt.txt','r', encoding='UTF8').read()
        
        if not self.memory:
            self.memory = self.load_history()
        # 1. 채팅 기록 불러오기 : loaded_memory 부분
        # 기록있으면 불러오고 없으면 비어있는 ConversationBufferMemory 생성

        loaded_memory = RunnablePassthrough.assign(
            chat_history=RunnableLambda(self.memory.load_memory_variables) | itemgetter("history"),
        )
        
        
        #2. 채팅기록과 현재 입력으로 새로운 입력 생성  : standalone_question 부분
        # chat_history와 현재 question을 이용해 질문 생성하는 템플릿
        _template = """Given the following conversation and a follow up Input, rephrase the follow up Input to be a standalone Input, in its original language.

        Chat History: {chat_history}
        Follow Up Input: {question}
        
        Standalone question:"""
        CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)

        # Now we calculate the standalone question
        standalone_question = {
            "standalone_question": {
                "question": lambda x: x["question"],
                "chat_history": lambda x: get_buffer_string(x["chat_history"]),
            }
            | CONDENSE_QUESTION_PROMPT
            | ChatOpenAI(temperature=0)
            | StrOutputParser(),
        }
        
        #3. 벡터DB에서 불러오기 : retrieved_documents 부분
        vectorstore = FAISS.load_local(self.database_path, embeddings=OpenAIEmbeddings()) #./data/database/faiss_index
        retriever = vectorstore.as_retriever()#(search_kwargs={"k": 50})

        retriever_from_llm = MultiQueryRetriever.from_llm(
            retriever=retriever, llm=ChatOpenAI(temperature=0),
        )
        # Now we retrieve the documents
        retrieved_documents = {
            "docs": itemgetter("standalone_question") | retriever_from_llm,
            "question": lambda x: x["standalone_question"],
        }


        #4. 최종 답하는 부분 : answer 부분
        # context를 참조해 한국어로 질문에 답변하는 템플릿
        template = self.ANSWER_PROMPT
        ANSWER_PROMPT = ChatPromptTemplate.from_template(template)

        DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template=self.DOCUMENT_PROMPT)

        def _combine_documents(docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"):
            doc_strings = [format_document(doc, document_prompt) for doc in docs]
            return document_separator.join(doc_strings)

        # Now we construct the inputs for the final prompt
        final_inputs = {
            "context": lambda x: _combine_documents(x["docs"]),
            "question": itemgetter("question"),
        }
        # And finally, we do the part that returns the answers
        answer = {
            "answer": final_inputs | ANSWER_PROMPT | ChatOpenAI(),#streaming=True,callbacks=[stream_it]),    #streaming?
            #"docs": itemgetter("docs"),
        }
        
        #5. 체인 연결
        # And now we put it all together!
        final_chain = loaded_memory | standalone_question | retrieved_documents | answer
        
        return final_chain
    
    def conversation_json(self):
        if not self.memory:
            self.memory = self.load_history()
        temp = self.memory.load_memory_variables({})['history']
        n=len(temp)//2
        d={'n': n, 'conversation':[]}
        for i in range(n):
            d['conversation'].append({'question':temp[2*i].content,'answer': temp[2*i+1].content})
        #j = json.dumps(d,ensure_ascii=False, indent=3)
        return d

    def memory_load_k(self, k:int):
        if not self.memory:
            self.memory = self.load_history()
        temp = self.memory.load_memory_variables({})['history']
        N_con = len(temp)//2
        if k >= N_con:
            return self.memory
        else:
            memory_k = ConversationBufferMemory(return_messages=True, output_key="answer", input_key="question")
            for i in range(N_con-k,N_con):
                memory_k.save_context({"question": temp[2*i].content},{"answer": temp[2*i+1].content})
            
            return memory_k