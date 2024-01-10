import pyrootutils
pyrootutils.setup_root(search_from = __file__,
                       indicator   = "README.md",
                       pythonpath  = True)

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

from project.server.modules.set_template import SetTemplate

# TODO: 리뷰 8개만 참고함. 임베딩 시 또는 훑을 때 어떻게 되는지 봐야함

class ChainPipeline():
    
    def __init__(self, 
                 user_id:str, 
                 keyword:str):
        self.BASE_DIR       = Path(__file__).parent.parent.parent / 'user_data' / user_id 
        self.history_path   = self.BASE_DIR / 'history' / keyword / f'{keyword}.pkl'
        self.database_path  = self.BASE_DIR / 'database' / keyword
        self.memory         = None
        self.user_id        = user_id
        self.keyword        = keyword
        self.stream_history = None
        self.config         = SetTemplate(user_id).load('chatgpt','conversation')
    
    def load_history(self):
        if self.history_path.is_file():
            with open(self.history_path,'rb') as f: #경로예시 : ./data/history/m.txt
                memory = pickle.load(f)
        else:
            memory = ConversationBufferMemory(
                return_messages = True, 
                output_key      = "answer", 
                input_key       = "question"
                )
            
        self.memory = memory
        
        return memory
    
    
    def save_history(self):
        if self.history_path.is_file() == False:
            os.makedirs(self.history_path.parent, exist_ok=True)
            
        with open(self.history_path,'wb') as f:
            pickle.dump(self.memory,f)


    def load_chain(self):
        chain_path = self.BASE_DIR / 'template' / 'chatgpt' 
        
        if not self.memory:
            self.memory = self.load_history()
        # 1. 채팅 기록 불러오기 : loaded_memory 부분
        # 기록있으면 불러오고 없으면 비어있는 ConversationBufferMemory 생성
        memory_k = self.memory_load_k(5)
        
        loaded_memory = RunnablePassthrough.assign(
            chat_history=RunnableLambda(memory_k.load_memory_variables) | itemgetter("history"),
        )
        #print(memory_k.load_memory_variables({}))
        #print(len(memory_k.load_memory_variables({})['history']))
        
        if self.config.condense == '':
            condense_prompt = self.config.condense_default
        else:
            condense_prompt = self.config.condense
            
        print(condense_prompt)    

        CONDENSE_QUESTION_PROMPT = ChatPromptTemplate.from_messages([
            ("system", condense_prompt + ' conversation : {chat_history}'),
            ("human", "{question}"),
        ])

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
        vectorstore = FAISS.load_local(folder_path = self.database_path, 
                                       embeddings  = OpenAIEmbeddings()) #./data/database/faiss_index
        retriever = vectorstore.as_retriever()#(search_kwargs={"k": 50})

        retriever_from_llm = MultiQueryRetriever.from_llm(
            retriever = retriever, 
            llm       = ChatOpenAI(temperature = 0,
                                   #model       = self.params.model),)
            ))
        # Now we retrieve the documents
        retrieved_documents = {
            "docs": itemgetter("standalone_question") | retriever_from_llm,
            "question": lambda x: x["standalone_question"],
        }

        #4. 최종 답하는 부분 : answer 부분
        # context를 참조해 한국어로 질문에 답변하는 템플릿
        
        if self.config.system == '':
            answer_prompt = self.config.system_default
        else:
            answer_prompt = self.config.system
        
        print(answer_prompt)
        
        ANSWER_PROMPT = ChatPromptTemplate.from_messages([
            ("system", answer_prompt),
            ("human", "{question}"),
        ])
        
        if self.config.document == '':
            document_prompt = self.config.document_default
        else:
            document_prompt = self.config.document

        DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template=document_prompt)
        print(DEFAULT_DOCUMENT_PROMPT)

        def _combine_documents(docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"):
            doc_strings = [format_document(doc, document_prompt) for doc in docs]
            print(doc_strings)
            return document_separator.join(doc_strings)

        # Now we construct the inputs for the final prompt
        final_inputs = {
            "context": lambda x: _combine_documents(x["docs"]),
            "question": itemgetter("question"),
        }
        # And finally, we do the part that returns the answers
        answer = {
            "answer": final_inputs | ANSWER_PROMPT | ChatOpenAI(),
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
        conversation = {'n': n, 'conversation':[]}
        
        for i in range(n):
            conversation['conversation'].append({'history_id': f'{self.user_id}_{self.keyword}_{i}',
                                                 'question':temp[2*i].content,
                                                 'answer': temp[2*i+1].content
                                                 })
        return conversation


    def memory_load_k(self, k:int):
        if not self.memory:
            self.memory = self.load_history()
            
        temp = self.memory.load_memory_variables({})['history']
        #print(temp)
        N_con = len(temp)//2
        
        if k >= N_con:
            return self.memory
        else:
            memory_k = ConversationBufferMemory(return_messages = True, 
                                                output_key      = "answer", 
                                                input_key       = "question")
            for i in range(N_con-k, N_con):
                memory_k.save_context({"question": temp[2 * i].content},
                                      {"answer": temp[2 * i+1].content})
            
            return memory_k
        
        
    async def streaming(self, chain, query):
        self.stream_history=''
        async for stream in chain.astream(query):
            self.stream_history += stream['answer'].content
            #print(self.stream_history)
            yield stream['answer'].content
        self.memory.save_context({"question" : query['question']}, {"answer" : self.stream_history})
        self.save_history()
        #print({"question" : query['question'], "answer" : self.stream_history})
    
class ReportChainPipeline():
        
    def __init__(self, 
                user_id:str, 
                keyword:str,
                report_template:str,
                document_template:str):
        self.BASE_DIR       = Path(__file__).parent.parent.parent / 'user_data' / user_id 
        self.database_path  = self.BASE_DIR / 'database' / keyword
        self.user_id        = user_id
        self.keyword        = keyword
        self.config         = SetTemplate(user_id)
        self.report_template = report_template
        self.document_template = document_template
    
    def load_chain(self):
        
        vectorstore = FAISS.load_local(folder_path = self.database_path, 
                                    embeddings  = OpenAIEmbeddings())
        retriever = vectorstore.as_retriever()#(search_kwargs={"k": 50})

        retriever_from_llm = MultiQueryRetriever.from_llm(
            retriever = retriever, 
            llm       = ChatOpenAI(temperature = 0,
                                #model       = self.params.model),)
            ))
        # Now we retrieve the documents
        retrieved_documents = retriever_from_llm.get_relevant_documents(query=self.report_template)

        # DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template=self.config.load_template('chatgpt','document'))
        DEFAULT_DOCUMENT_PROMPT = PromptTemplate.from_template(template=self.document_template)

        def _combine_documents(docs, document_prompt=DEFAULT_DOCUMENT_PROMPT, document_separator="\n\n"):
            doc_strings = [format_document(doc, document_prompt) for doc in docs]
            return document_separator.join(doc_strings)
        
        ANSWER_PROMPT = self.report_template.format(context = _combine_documents(retrieved_documents))
        #print(ANSWER_PROMPT)
        self.save_template()
        return ChatOpenAI().predict(ANSWER_PROMPT)
    
    def save_template(self):
        #if self.config.load_template('chatgpt','report')[:-1] == self.report_template:
        config = self.config.params.load(self.BASE_DIR / 'template' / 'configs.yaml' ,addict=False)
        #print(config['chatgpt']['templates']['report']['prompt'])
        #print(config['chatgpt']['templates']['report']['document'])
        # print(config['chatgpt']['templates']['report'])
        config['chatgpt']['templates']['report']['prompt'] = self.report_template
        #if self.config.load_template('chatgpt','document')[:-1] == self.document_template:
        config['chatgpt']['templates']['report']['document'] = self.document_template
        # print(config)
        # print(self.config.base_save_dir / 'configs.yaml')
        self.config.params._save(config, self.config.base_save_dir , 'configs.yaml')
        # print(self.config.load_template('chatgpt','document')[:-1],len(self.config.load_template('chatgpt','document')))
        # print(self.document_template,len(self.document_template))