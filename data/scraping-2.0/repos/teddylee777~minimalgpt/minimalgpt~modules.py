from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationChain
from langchain.chains.llm import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType


class BaseModule():
    def __init__(self, streaming=True, **model_kwargs):
        self.model_kwargs = self.create_model_kwargs(streaming, **model_kwargs)
            
    def create_model_kwargs(self, streaming=True, **model_kwargs):
        # 객체 생성
        if streaming:
            model_kwargs['streaming'] = True
            model_kwargs['callbacks'] = [StreamingStdOutCallbackHandler()]
        return model_kwargs

class ChatModule(BaseModule):
    def __init__(self, streaming=True, **model_kwargs):
        super().__init__(streaming, **model_kwargs)
        self.llm = ChatOpenAI(**self.model_kwargs)
        
    def ask(self, question):
        return self.llm.predict(question)


class ConversationModule(BaseModule):
    def __init__(self, streaming=True, **model_kwargs):
        super().__init__(streaming, **model_kwargs)
        llm = ChatOpenAI(**self.model_kwargs)
        self.conversation = ConversationChain(llm=llm)
        
    def ask(self, question):
        return self.conversation.run(question)
    
    
class WebSummarizeModule(BaseModule):
    def __init__(self, url, mode='stuff', streaming=False, **model_kwargs):
        super().__init__(streaming, **model_kwargs)
        self.mode = mode
        # 웹 문서 크롤링
        self.loader = WebBaseLoader(url)
        
        if mode == 'stuff':
            self.docs = self.loader.load()
        else:
            # 뉴스기사의 본문을 Chunk 단위로 쪼갬
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=6000, 
                                                    chunk_overlap=50, 
                                                    separators=["\n\n", "\n", "(?<=\. )", " ", ""],
                                                    length_function=len,
                                                    )
            
            # 웹사이트 내용 크롤링 후 Chunk 단위로 분할
            self.docs = WebBaseLoader(url).load_and_split(text_splitter)
        
        self.llm = ChatOpenAI(**self.model_kwargs)
        
    def ask(self, combine_template):
        if self.mode == 'stuff':
            combine_prompt = PromptTemplate.from_template(combine_template)
            llm_chain = LLMChain(llm=self.llm, prompt=combine_prompt)
            stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")
            
            return stuff_chain.run(self.docs)
        else:
            # 각 Chunk 단위의 템플릿
            template = '''다음은 웹사이트에 수집한 정보의 일부분입니다. 핵심 내용을 간략하게 요약해 주세요. 다른 언어로 번역하지 말아주세요. 본문에 없는 내용을 언급하지 말아 주세요.:

            {text}
            '''

            # 템플릿 생성
            prompt = PromptTemplate(template=template, input_variables=['text'])
            combine_prompt = PromptTemplate(template=combine_template, input_variables=['text'])

            # 요약을 도와주는 load_summarize_chain
            chain = load_summarize_chain(self.llm, 
                                        map_prompt=prompt, 
                                        combine_prompt=combine_prompt, 
                                        chain_type="map_reduce", 
                                        verbose=False)
            
            return chain.run(self.docs)


class PandasModule(BaseModule):
    def __init__(self, df, streaming=False, **model_kwargs):
        super().__init__(streaming, **model_kwargs)
        llm = ChatOpenAI(**self.model_kwargs)
        # 에이전트 생성
        self.agent = create_pandas_dataframe_agent(
            llm,                                   # 모델 정의
            df,                                    # 데이터프레임
            verbose=True,                          # 추론과정 출력
            agent_type=AgentType.OPENAI_FUNCTIONS, # AgentType.ZERO_SHOT_REACT_DESCRIPTION
        )
        
    def ask(self, query):
        return self.agent.run(query)

