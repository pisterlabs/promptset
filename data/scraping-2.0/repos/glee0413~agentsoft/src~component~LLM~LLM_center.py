from langchain.chat_models import QianfanChatEndpoint
from langchain.schema import HumanMessage
from pydantic_settings import BaseSettings

class BaiduConfig(BaseSettings):
    QIANFAN_AK: str
    QIANFAN_SK: str
    
    class Config:
        env_file = ".env_baidu"

class LLMCenter():
    def __init__(self) -> None:
        self.baidu_config = BaiduConfig()
        pass
    
    async def agentrate(message:str):
        chatLLM = QianfanChatEndpoint(streaming=True,)
        # res = chatLLM.stream(streaming=True)
        # for r in res:
        #     print(f"chat resp: {r}\n")
        
        async def run_aio_generate():
            resp = await chatLLM.agenerate(
                messages=[[HumanMessage(content=message)]]
            )
            print(resp)
        print('call run_aio_generate')
        await run_aio_generate()