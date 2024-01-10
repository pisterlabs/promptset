import os
import pandas as pd
from pathlib import Path
from fastapi import APIRouter
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from langchain.embeddings import OpenAIEmbeddings

from server.modules.set_template import SetTemplate
from llm_model.llama2_answer import LangchainPipline
from server.modules.crawl_pipeline import CrawlManager
from server.modules.chain_pipeline import ChainPipeline,ReportChainPipeline
from server.modules.vectordb_pipeline import VectorPipeline
from fastapi.responses import FileResponse

class Quest(BaseModel):
    question: str

class Report(BaseModel): #report에 적용
    user_id: str
    keyword: str

class Template(BaseModel):
    template_config: dict[str, str]

class UserOut(BaseModel):
    answer: str
    
class Topic(BaseModel):
    keyword: str

class FastApiServer:
    
    def __init__(self):
        
        self.router = APIRouter()
        self.register_routes()
        
    def register_routes(self):
        
        self.router.add_api_route("/", self.chat_list, methods=["GET"])
        self.router.add_api_route("/history/{user_id}/{keyword}", self.history, methods=["GET"])
        self.router.add_api_route("/answer/{user_id}/{keyword}/{stream}", self.answer, methods=["POST"])
        
        self.router.add_api_route("/start_crawl/{user_id}/{keyword}", self.start_crawl, methods=["GET"])
        self.router.add_api_route("/crawl_data/{user_id}/{keyword}", self.get_crawl_data, methods=["POST"])
        
        self.router.add_api_route("/vectordb/{user_id}/{method}/{keyword}", self.manage_vectordb, methods=["GET"])
        self.router.add_api_route("/new_chat/{user_id}", self.new_chat, methods=["GET"])
        self.router.add_api_route("/report", self.report, methods=["POST"])
                
        self.router.add_api_route("/load_template/{user_id}/{llm}/{template_type}", self.load_template, methods=["GET"])
        self.router.add_api_route("/edit_template/{user_id}/{llm}/{template_type}", self.edit_template, methods=["POST"])
        
        self.router.add_api_route("/llama/{user_id}/{question}", self.llama_answer, methods=["GET"])

    async def chat_list(self, user_id: str):
        
        chat_list_path = Path(__file__).parent.parent / 'user_data' / user_id / 'database'
        chatlist = os.listdir(chat_list_path)
        
        return chatlist
    
    async def answer(self,
                     user_id: str,
                     keyword: str,  
                     stream: bool,
                     item: Quest):
        
        chainpipe       = ChainPipeline(user_id = user_id, 
                                        keyword = keyword)
        history         = chainpipe.load_history()
        chain           = chainpipe.load_chain()
        response_input  = {'question': item.question}
        
        if stream == True:
            return StreamingResponse(content    = chainpipe.streaming(chain, response_input), 
                                     media_type = "text/event-stream")
        else:
            result = chain.invoke(response_input)
            history.save_context(response_input, {"answer" : result["answer"].content})
            chainpipe.memory = history
            chainpipe.save_history()
            return result
    
    async def report(self, data: Report):
        chainpipe = ReportChainPipeline(data.user_id, data.keyword)
        result = chainpipe.load_chain()
        return FileResponse(path = result, filename='test.pdf', media_type='application/octet-stream')
        #return result
    
    async def history(self, 
                      user_id:str, 
                      keyword:str):
        
        chainpipe = ChainPipeline(user_id = user_id,
                                  keyword = keyword)
        history_conversation = chainpipe.conversation_json()
        
        return history_conversation
    
    async def load_template(self,
                            llm:str,
                            user_id:str,
                            template_type:str,):
        
        st = SetTemplate(user_id=user_id)
        template = st.load(llm           = llm, 
                           template_type = template_type)
        
        return template
        
    
    async def edit_template(self,
                            llm:str,
                            user_id:str,
                            template_type:str,
                            config:Template):
        
        st = SetTemplate(user_id=user_id)
        
        st.edit(llm           = llm,
                template_type = template_type,
                **config.template_config)
        
        
    def llama_answer(self, 
                     user_id,
                     question,
                     ): # 임시. 테스트로 두고 크롤링 파이프라인 개발하면 삭제할 예정
        
        lp = LangchainPipline(user_id=user_id)
        result = lp.chain(question=question).strip()
        
        return result
    
    
    def manage_vectordb(self, 
                        user_id: str, 
                        method,  # url에 method를 넣는게 아니라 http통신에서 method를 통해 가져오기
                        keyword: str):

        if method == 'delete':
            
            return VectorPipeline.delete_store_by_keyword(user_id = user_id,
                                                          keyword = keyword)

                    
    async def start_crawl(self,
                          user_id: str,
                          keyword: str):
        
        target_col = 'document'
        
        lp = LangchainPipline(user_id=user_id) # 모델 미리 load
        
        ### 크롤링 시 redis를 활용한 메세지큐 구현 
        
        cm = CrawlManager(user_id=user_id,
                          keyword=keyword)
        cm.run()
        
        data = pd.read_csv(cm.base_dir / 'merged_data.csv')
        
        filtered_data = []
        for _, row in data.iterrows():
            if lp.chain(question = row[target_col]).strip() == 'Yes':
                filtered_data.append(row)

        result_df = pd.DataFrame(filtered_data)
        result_df.to_csv(cm.base_dir / 'filtered_data.csv', index=False)
        
        VectorPipeline.embedding_and_store(data       = data,
                                           user_id    = user_id,
                                           keyword    = keyword,
                                           target_col = target_col,
                                           embedding  = OpenAIEmbeddings(),
                                           )
      
    
    async def new_chat(self, user_id):
        template = SetTemplate(user_id=user_id)
        template.set_initial_templates()
        
        return {'status':'new_chat created!'}
    
    async def get_crawl_data(self,
                             user_id: str, 
                             keyword:str):

        cm = CrawlManager(user_id=user_id,
                          keyword=keyword)
        
        try:
            return cm.get_crawl_data()
        except:
            return {'status':False}
     
        
        
        