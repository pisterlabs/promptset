import streamlit as st
import pandas as pd

import openai
import streamlit as st
from streamlit_chat import message
from streamlit_autorefresh import st_autorefresh

import json
import data_tools

import dotenv
import os

def draw():
    config = dotenv.find_dotenv()
    dotenv.load_dotenv(config)
    openai.api_key = os.environ["OPENAI_API_KEY"]
    st.header("🤖 OPT-3 (Demo)")
    
    #### 기본 변수 공간 선언 ####
        
    if 'generated' not in st.session_state:
        st.session_state['generated'] = []
     
    if 'past' not in st.session_state:
        st.session_state['past'] = []
    ##################
    
    
    def generate_response(user_input): # gpt로 답변 만들기
        prompt = "You are a counselor for Ottogi(한국어로 오뚜기) food company. Talk shorter than normal."\
            "Below things are history of your chat"\
            f"{st.session_state['generated']}"\
            f"{st.session_state['past']}"
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  
        # model="gpt-4-0613",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input},
        ],
        # max_token=200,
        # stop=None,
        # temperature=0,
        # top_p=1,
        )
     
        message = response.choices[0].message.content
        st.session_state.past.append(user_input)
        st.session_state.generated.append(message)
        
    def Chat_message(generated, key): # 답변 설정
        message(generated, 
        key=key, 
        avatar_style="initials", # 아이콘 변경시 오른쪽 문서를 참고할 것 https://docs.streamlit.io/library/api-reference/chat/st.chat_message
        seed="🤖")
        
    def Make_output(user_input):
        newinput_check = 0
        ## Manual part
        if user_input == "1. Discount News":
            dis_num = 5
            output = f"## Here is TOP {dis_num} discount products!"
            
            sku_id = list(data_tools.discount_level.keys())[:dis_num]
            for i in sku_id:
                product_name = data_tools.sku_to_name(i)
                product_url = data_tools.sku_to_purchase_url(i)
                # output += f'\n- <a href="{product_url}">{product_name}</a>'
                output += f'\n- [{product_name}]({product_url})'
        
        elif user_input == "2. Best-Seller Foods":
            dis_num = 5
            output = f"## Here is TOP {dis_num} Best-seller foods!"
            
            sku_id = list(data_tools.order_rank.keys())[:dis_num]
            for i in sku_id:
                product_name = data_tools.sku_to_name(i)
                product_url = data_tools.sku_to_purchase_url(i)
                output += f'\n- [{product_name}]({product_url})'
                
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)
        
    ## Intro Part
    
    if st.session_state['generated'] == []:
        output = "Hello! I'm Ottogi Intern, OPT! I can recommand you about \n 1. discount news \n 2. Best-seller foods"
        st.session_state.generated.append(output)
    
    
    
    
    
    ## 답변 전송란
    with st.form('form', clear_on_submit=True):
        
        Intro = "Ask to OPT!"
            
        user_input = st.text_input(f'{Intro}: ', '', key='input')
        submitted = st.form_submit_button('Send') # form 아이콘 변경 할 것
    
    ## generated handle part
    if user_input and submitted:
        generate_response(user_input)
        
    ## Expected Questions made by boxes
    col1, col2 = st.columns([0.1, 0.4], gap="small")
    with col1:
        if st.button("1. Discount News", type="secondary"):
            Make_output("1. Discount News")
    with col2:
        if st.button("2. Best-Seller Foods", type="secondary"):
            Make_output("2. Best-Seller Foods")
    
    ## Display Part
    if st.session_state['generated']:
        for i in range(len(st.session_state['generated'])-2, -1, -1):
            Chat_message(st.session_state["generated"][i+1], str(i+1))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        Chat_message(st.session_state["generated"][0], 'start_use')
    
    # 검증용 코드
    # if st.session_state['generated']:
        
    #     st.text("{st.session_state['generated']}")
    #     st.text(st.session_state['past'])