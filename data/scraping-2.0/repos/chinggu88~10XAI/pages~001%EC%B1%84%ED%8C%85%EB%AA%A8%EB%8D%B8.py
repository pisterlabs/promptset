import streamlit as st
from llm.ai import aihelp
from langchain.memory import StreamlitChatMessageHistory

from llm.customai import customai


if "step" not in st.session_state:
        st.session_state["step"] = 0
step = st.session_state["step"]
if step == 0:
    """
    Insert your database url only mysql, mariadb 
    ex) {mysql+pymysql://user:password@ip:port/dbname}
    """
    ip = st.text_input('IP ex: 127.0.0.1')
    port = st.text_input('PORT ex: 3306')
    user = st.text_input('USER ex: user')
    pw = st.text_input('PASSWORD ex: 1234567')
    db = st.text_input('PORT ex: yout database name')
    
    bt = st.button('Connect Database',type="primary")
    if bt:
        if ip == '':
            'input your ip'
        elif port == '':
            'input your port'
        elif user == '':
            'input your user' 
        elif pw == '':
            'input your pw'
        elif db == '':
            'input your db'
        else :
            cai = customai(ip,port,user,pw,db)
            status = cai.getconnect()
            if status:
                status
                st.session_state["ip"] = ip
                st.session_state["port"] = port
                st.session_state["user"] = user
                st.session_state["pw"] = pw
                st.session_state["db"] = db
                st.session_state["step"] = 1
            else:
                st.header('check your databse info')

               
                

if step == 1:
    history = StreamlitChatMessageHistory(key="chat_messages")
    ip =st.session_state["ip"] 
    port =st.session_state["port"] 
    user =st.session_state["user"]
    pw = st.session_state["pw"] 
    db =st.session_state["db"]
    ai = customai(ip,port,user,pw,db)
    status  = ai.getconnect()
    st.title(status)
    # if "msg" not in st.session_state:
    #     st.session_state["msg"] =[]
        

    # def send_msg(msg,role,save = True):
    #     with st.chat_message(role):
    #         st.write(msg)
    #     #저장
    #     if save:
    #         st.session_state["msg"].append({"msg":msg,"role":role})
        
    # for msg in st.session_state["msg"]:
    #     send_msg(msg["msg"],msg['role'],False) 
        

    # msg = st.chat_input('대화를 입력하세요')

    # if msg:
    #     send_msg(msg,"human")
    #     #캐쉬작업 추가
    #     history.add_user_message(msg)
    #     with st.spinner("물어보는중"):
    #         #ai에 묻기
    #         data = ai.messageai(msg)
    #         print(data)
    #         history.add_ai_message(data)
    #         # send_msg(data,"ai")
    #         for i in range(len(data)):
    #             send_msg(data[i],"ai")



    
        