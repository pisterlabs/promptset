# -*- coding: utf-8 -*- 
import gradio as gr
import os

from pathlib import Path
import json
from konlpy.tag import Okt

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory

class ChatBot():

    # 생성자
    def __init__(self) -> None:

        # 프롬프트 
        self.prompt = self.get_prompt()
        # 모델
        self.model = self.get_model()
        # 형태소 분석기
        self.okt = Okt()
    
    # 모델 로드
    def get_model(self) -> ChatOpenAI:

        with open('resource/secret.json', 'r') as json_file:
            secret_json = json.load(json_file)
            
        openai_api_key = secret_json["OPENAI_API_KEY"]
        
        # 1) GPT 3.5 (*권장)
        defalut_model = "gpt-3.5-turbo"
        # 2) GPT 3.5 Fine-tunning (custom데이터 10개 학습)
        fine_tunning_model = secret_json["FINE_TUNNING_MODEL"]
        
        chat_model = ChatOpenAI(model=defalut_model, openai_api_key=openai_api_key)
        conversation = ConversationChain(
            
            # 프롬프트 템플릿 적용
            prompt=self.prompt,
            # 모델 적용
            llm=chat_model,
            verbose=False,
            # 대화 내용 기억 (k번까지)
            memory=ConversationBufferWindowMemory(memory_key='history', ai_prefix="AI Assistant", k=5)
        )
        return conversation
                #     만약 사용자의 질문이 "목배게 추천해줘" 와 같은 의미라면, "목배게 추천해드리겠습니다!"라고 말해줘. 이와 유사하게 "허리에 좋은 의자좀 알려줘"라고 질문한다면
                # "의자 추천해드리겠습니다."라고 말해줘. 또 유사하게 "손목에 좋은 마우스 추천해봐"라고 한다면 "마우스 추천해드리겠습니다!"라고 말해줘.

    # 프롬프트 템플릿
    def get_prompt(self) -> ChatPromptTemplate:
        template = """
                너는 '척추적 60분' 이라는 자세 교정 서비스의 챗봇이야.

                너가 최우선으로 할일은 사용자의 질문(입력값)이 상품을 추천해 달라는 의미인지 아닌지 분류하는 것이야. 사용자는 "목배게 추천해줘", "허리에 좋은 의자좀 알려줘" ,"손목에 좋은 마우스 추천해줄래?","쓸만한 모니터 거치대좀 알려줘봐"와 같이 상품을 추천해달라는 대화를 입력할 수도 있어. 이때는 사용자가 원하는 물품+"추천해드리겠습니다!"라고 한문장으로만 답변해줘.

                만약 사용자의 질문(입력값)이 상품을 추천해달라는 의미가 아니라면 '척추적 60분' 서비스 사용법 안내나 스트레칭 방법에 대한 질문이 입력될 수 있어. 이때 다음의 내용을 참고하여 답변해주어야해. '척추적 60분' 서비스는 컴퓨터 앞에서 공부하거나 작업할 때 사용하는 자세 교정 서비스야. 노트북 웹캠을 통해  실시간으로 입력되는 영상으로 앉은 자세를 감지하면서 구부정한 자세나 거북목, 졸고 있을 때를 유저에게 알려주는 서비스야. '척추적 60분'의 기능/세부사항을 구체적으로 알려줄께.  첫번째로  '요정만들기' 기능은 감지된 정자세/비대칭 자세 비율을 토대로 직선 혹은 기울어진 척추 모형을 쌓고 귀여운 척추 캐릭터(척추 요정)을 만드는 컨텐츠라고 생각하면 돼. 완성된 척추 캐릭터들은 마이페이지에서 일간/주간/월간 단위로 확인할 수 있고 자신의 자세가 어땠는지 알아볼 수 있어. 두번째로, 척추 요정(척추 캐릭터)를 만들기 위해 활용되는  척추(모형)은 서비스를 사용하는 유저의 자세에 따라 30분단위로 생성돼. 30분동안 구부정한 자세로 작업했다면 기울어진 척추 모형이 생성되고 올바른 자세로 작업했다면 올곧은 모형이 생겨. 세번째로,  최초 프로그램(서비스) 사용 시 유저들의 정자세 사진 1장을 촬영할 꺼야. 그 이유로 구부정한자세/거북목/졸음을 감지하고 구분하기 위해서 올바른 자세로 앉아 있는 사진이 필요하기 때문이야. 이 사진은 개인정보이기 때문에 이와 다른 용도로는 활용하지 않을꺼야.  네번째로, 유저들은 컴퓨터 앞에 앉아 작업하다가 잠시 화장실에 가거나 다른 업무를 위해 자리를 비울 수 있어. 이때는 화면 좌측 상단의 '일시정지' 버튼을 눌러 자세 감지를 잠시 중단할 수 있어. 이 내용들을 바탕으로 답변해주면 돼.

                질문에 대해 답변할 때는 이모티콘을 많이 활용하고 가장 친한 친구 말투로 말해주면 좋겠어. 

                너의 이름은 스켈레톤의 '레톤'을 딴 '레톤이'야.  

                모든 답변은 반드시 20자 이내로 간결하게 대답해야해.
                Current conversation:
                {history}
                Human: {input}
                AI Assistant:"""

        prompt = PromptTemplate(input_variables=["history","input"], template=template)
        return prompt

    # 답변 추론
    def get_answer(self, text) -> str:
        return self.model.predict(input=text)


# 테스트 Main 함수        
if __name__ == '__main__':
    chatbot = ChatBot()

    with gr.Blocks() as demo:
        gr_chatbot = gr.Chatbot()
        msg = gr.Textbox()
        clear = gr.Button("Clear")

        def respond(message, chat_history):
            # ChatGPT 질의
            bot_message = chatbot.get_answer(message)
            # 답변 기록
            chat_history.append((message, bot_message))
            return "", chat_history

        msg.submit(respond, [msg, gr_chatbot], [msg, gr_chatbot])
        clear.click(lambda: None, None, gr_chatbot, queue=False)

    # 로컬(local) 구동시
    demo.launch(share=True)