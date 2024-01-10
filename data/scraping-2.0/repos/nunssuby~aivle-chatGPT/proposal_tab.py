import streamlit as st
import openai

# 동적 화면 구성시 마땅한 아이디어가 안 떠올라서 sesstion_state를 사용했더니 비동기 callback 지옥처럼 변수지옥에 빠졌음
if 'chatGPT_callback' not in st.session_state:
    st.session_state['chatGPT_callback'] = ''  
if '1.1' not in st.session_state:
    st.session_state['1.1'] = ''
if '1.1_callback' not in st.session_state:
    st.session_state['1.1_callback'] = False
if '2.1' not in st.session_state:
    st.session_state['2.1'] = ''
if '2.1_callback' not in st.session_state:
    st.session_state['2.1_callback'] = False
if '2.2' not in st.session_state:
    st.session_state['2.2'] = '' 
if '2.2_callback' not in st.session_state:
    st.session_state['2.2_callback'] = False
if '3.1' not in st.session_state:
    st.session_state['3.1'] = '' 
if '3.1_callback' not in st.session_state:
    st.session_state['3.1_callback'] = False    
if '3.2' not in st.session_state:
    st.session_state['3.2'] = ''
if '3.2_callback' not in st.session_state:
    st.session_state['3.2_callback'] = False
if '3.3' not in st.session_state:
    st.session_state['3.3'] = '' 
if '3.3_callback' not in st.session_state:
    st.session_state['3.3_callback'] = False     
if '4.1' not in st.session_state:
    st.session_state['4.1'] = ''           
if '4.1_callback' not in st.session_state:
    st.session_state['4.1_callback'] = False            
if '4.2' not in st.session_state:
    st.session_state['4.2'] = '' 
if '4.2_callback' not in st.session_state:
    st.session_state['4.2_callback'] = False
if '5.1' not in st.session_state:
    st.session_state['5.1'] = ''            
if '5.1_callback' not in st.session_state:
    st.session_state['5.1_callback'] = False           
if '5.2' not in st.session_state:
    st.session_state['5.2'] = '' 
if '5.2_callback' not in st.session_state:
    st.session_state['5.2_callback'] = False
if 'chatGPTCount' not in st.session_state:
    st.session_state['chatGPTCount'] = 0     
if 'chatGPTRequestCount' not in st.session_state:
    st.session_state['chatGPTRequestCount'] = 0       
openai.api_key = "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"
# chatGPT API를 모듈화
def chatGPT(content,session_state_no):
    callback_name = session_state_no+'_callback' 
    st.session_state['chatGPTCount']=st.session_state['chatGPTCount']+1
    print("chatGPT호출 횟수 : ",st.session_state['chatGPTCount'])
    print("chatGPT호출 session_state_no  : ",session_state_no)
    print("chatGPT호출 내용  : ",content)
    # st.session_state['chatGPT_callback'] = False
    st.session_state[callback_name] = False
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": content}]
    )
    print(completion)
    # st.session_state['chatGPT_callback'] = True
    st.session_state[callback_name] = True
    return completion

def chatGPTResponse(content,session_state_no):
    return chatGPT(content,session_state_no).choices[0].message.content


def whenInputRequestChatGPT(session_state):
    st.session_state['chatGPTRequestCount']=st.session_state['chatGPTRequestCount']+1
    print("chatGPTRequestCount 호출 횟수 : ",st.session_state['chatGPTRequestCount'])
    print("session_state : ",session_state)
    if(session_state):
        print("성공했을 경우 chatGPTRequestCount  : ",st.session_state['chatGPTRequestCount'])
        st.subheader(f'chatGPT의 응답: \r\n {chatGPTResponse(session_state)}')

def requestChatGPT(session_state_no):
    content = st.session_state[session_state_no]
    st.session_state['chatGPTRequestCount']=st.session_state['chatGPTRequestCount']+1
    print("chatGPTRequestCount 호출 횟수 : ",st.session_state['chatGPTRequestCount'])
    print("session_state_no : ",session_state_no)
    print("content : ",content)
    callback_name = session_state_no+'_callback' 
    if(content and st.session_state[callback_name]==False):
        print("성공했을 경우 chatGPTRequestCount  : ",st.session_state['chatGPTRequestCount'])
        st.subheader(f'chatGPT의 응답: \r\n {chatGPTResponse(content, session_state_no)}')        

tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs(["개요","1단계", "2단계", "3단계","4단계","5단계"])

with tab0:
    st.title('chatGPT 제안서 작성')
    st.write('FinanceData.KR - chatGPT API 활용를 참고했습니다.')
    st.write('전체 url : https://financedata.notion.site/ChatGPT-6517d755152d40a99ae360a97e83b24c')
    st.text('1단계부터 ~5단계까지 순서대로 진행해주시고, 중간에 오류 발생시 다음단계로 넘어가면 됩니다.')
with tab1:
    st.header ('1. 배경과 상황을 설명합니다')
    st.text('배경을 설명합니다. ')
    st.text('ChatGPT에게 텍스트 데이터를 제공(feed) 합니다. ')
    background_textarea_value_pre='다음과 같은 제안요청서를 받았어'
    background_textarea_value_post='일단 기억해줘'


    st.session_state['1.1'] = st.text_area(label="[제안요청서-RFP] 내용을 붙여넣어 주세요 \n 예시는 FinanceData.KR의 chatGPT API 활용을 참고하세요 ")

    if(st.session_state['1.1'] and st.session_state['1.1_callback']==False):
        print("1.1")
        content = f"{background_textarea_value_pre} \n {st.session_state['1.1']} \n {background_textarea_value_post}"
        st.write(content)
        st.subheader(f'chatGPT의 응답: \r\n {chatGPTResponse(content,"1.1")}')
        st.session_state['1.1_callback'] = True

    if(st.session_state['1.1'] and st.session_state['1.1_callback'] ):   
        print("1.1 and chatGPT_callback : " ) 
        st.write('위 내용 중에 어려운 내용은 별도로 chatGPT에게 물어보면서 알아두세요')
        st.write('1단계 완료 2단계로 넘어가세요')

    

with tab2:
    st.header ('2. 원하는 것이 무엇인지 설명합니다')
    st.text('원하는 것이 무엇인지(특히, 강조할 부분을) 명시 합니다. ')
    st.session_state['2.1'] = st.text_input(label="예) 나는 시스템 구축에 있어서 기술적 우위를 강조하는 경쟁력 있는 제안서를 작성하려고 하려고해")

    requestChatGPT('2.1')  

    if(st.session_state['2.1'] and st.session_state['2.1_callback'] ):  
        print("2.1 and chatGPT_callback : " ) 
        st.markdown(f"#### [링크의 목차를 참고하여 작성해보세요](https://financedata.notion.site/4c2e5df6e02942f1bdd11d92ed24e36f)") 
        st.session_state['2.2'] = st.text_area(label="목차를 넣어보세요")

    requestChatGPT('2.2')  
    if(st.session_state['2.2'] and st.session_state['2.2_callback'] ):
        st.write('2단계 완료 3단계로 넘어가세요')
with tab3:
    st.header ('3. 어떻게 작업을 진행할지 묻습니다')       
    st.text('진행의 방법과 단계를 질문하고, 가능한 자원(인력) 혹은 시간제한 등을 설명합니다.')
    st.session_state['3.1'] = st.text_input(label="예) 제안서 작성작업을 어떤 수순으로 진행하면 좋을지 단계별로 설명해줘. 나를 포함해서 3명이 작업을 할 예정이야.")

    requestChatGPT('3.1')

    if(st.session_state['3.1'] and st.session_state['3.1_callback'] ): 
        st.session_state['3.2'] = st.text_input(label="예) 하루안에 작업을 마치려고해 적절한 시간 배분을 해줘")

    requestChatGPT('3.2')
    if(st.session_state['3.2'] and st.session_state['3.2_callback'] ):    
        st.session_state['3.3'] = st.text_input(label="예) 최종 검토와 제출 단계는 제외해줘")    

    requestChatGPT('3.3')
    if(st.session_state['3.3'] and st.session_state['3.3_callback'] ):    
        st.write('3단계 완료 4단계로 넘어가세요')
with tab4:
    st.header ('4. 항목별로 자세하게 작성해 달라고 요청합니다') 
    st.text('하위 항목에 대해 자세하게 작성해다라고 요청합니다. ')
    st.text('구체적인 내용을 질문하거나, 재작성을 요청합니다.')
    st.text('요약하라고 요청합니다.')
    st.write('예시) ')
    st.text("다음을 작성해줘 \n 4. 사업 관리 부문 \n\t a. 프로젝트 수행 방법론 \n\t b. 프로젝트 관리 방안 \n\t c. 품질 관리 계획")
    st.session_state['4.1'] = st.text_area(label="하위 항목을 자세하게 해달라고 요청합니다.예시참고 ")
    requestChatGPT('4.1')
    if(st.session_state['4.1'] and st.session_state['4.1_callback'] ):      
        st.session_state['4.2'] = st.text_input(label="예) '4. 사업 관리' 부문의 '3. 품질 관리 계획' 부분을 요약하고, 모든 문장들이 명사로 끝나도록, 예를 들어 ~함, ~임 이런식의 문장이 되도록 다시 작성해줘.")    

    requestChatGPT('4.2')
    if(st.session_state['4.2'] and st.session_state['4.2_callback'] ):  
        st.write('4단계 완료 5단계로 넘어가세요')
with tab5:
    st.header ('5. 가이드라인 제시') 
    st.text('필요하면, ChatGPT에게 지식을 전달합니다.')
    st.text('중간 중간 요약하거나 리마인드 합니다. (대화가 길어지면 가끔 삼천포)')

    st.write('예시) ')
    st.text('사업 관리 부문 의 품질 관리 계획 부분에서 테스트 주도 개발(TDD) 방식을 적용하여 품질 관리를 강화한다고 언급을 했는데, \n 테스트 주도 개발(TDD) 방식이 어떻게 품질 관리를 강화에 도움이 되는지에대한 설명을 추가해서 품질 관리 계획 부분만 다시 작성해줘.')
    st.session_state['5.1'] = st.text_area(label="위 예시를 참고하여 질문하세요. ")

    requestChatGPT('5.1')
    if(st.session_state['5.1'] and st.session_state['5.1_callback'] ):  
        st.write('예시) ')
        st.text('TDD 방식을 적용하기 위해 프로젝트 초기 단계에서 각 기능의 정확한 요구사항을 정의하기 위한 구체적인 활동을 3가지 나열하고, 이 활동을 추가해서 사업 관리 부문을 다시 작성해줘')
        st.session_state['5.2'] = st.text_area(label="위 예시를 참고하여 추가 질문하세요. ")

    requestChatGPT('5.2')
    if(st.session_state['5.2'] and st.session_state['5.2_callback'] ):  
        st.header ('수고하셨습니다.')     
        st.header ('적절한 하위 부분들을 나누어 작성하고, 각 단계별 텍스트를 취합합니다.')  