import streamlit as st
from langchain.memory import StreamlitChatMessageHistory
from langchain.callbacks import get_openai_callback

from src.core.agent.electricity_agent import ElectricityAgent
from src.core.model.chatbot_model import OpenAiModel



# Field
chat_model = OpenAiModel()
st.set_page_config(
    page_title="CourtauctionAI",
    page_icon="🐢",
)
st.title("CourtauctionAI 🐢")

if "conversation" not in st.session_state:
    st.session_state.conversation = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = None

# Sidebar Create
with st.sidebar:
    uploaded_file_list = st.file_uploader("Upload a file", type=["pdf", "docx", "txt", "csv"], accept_multiple_files=True)
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    process = st.button("적용")

# process 버튼 클릭 시
if process:
    if not openai_api_key:
        # TODO - 삭제 필요
        openai_api_key = 'sk-03hkFJpTeBGlDDbtNSzql1EtUs'
        # st.info("Please enter your OpenAI API Key")
        # st.stop()

    # LLM 대신 chat_model.get_conversation_chain(openai_api_key) 적용
    # st.session_state.conversation = chat_model.get_conversation_chain(openai_api_key)
    st.session_state.conversation = ElectricityAgent(chat_model.get_conversation_chain(openai_api_key))
    if uploaded_file_list and len(uploaded_file_list) > 0:
        chat_model.load_file_list(uploaded_file_list)

# Messages가 session_state 내 없을 경우
if 'messages' not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "안녕하세요! 저는 CourtauctionAI 입니다. 어떤 도움이 필요하신가요?"}]

# 화면내 전체 Messages 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ###################################################### 실행부 ######################################################

def assistant_chat(query: str):
    # TODO - 주석 해제
    # if not st.session_state.conversation:
    #     st.info("Please Enter your OpenAI API Key")
    #     st.stop()
    # 응답 값 출력
    with st.chat_message("assistant"):
        chain = st.session_state.conversation
        with st.spinner("답변을 생성중입니다..."):
            # 유저 질문 질의
            # result = chain({"question": query})
            result = chain.run(query, ["1000000001", "1000000002"])
            with get_openai_callback() as cb:
                st.session_state.chat_history = result["chat_history"]
            # 응답 결과 출력
            response = result["answer"]
            st.markdown(response)
            # 참고 문서 확인
            with st.expander("참고 문서 확인"):
                source_documents = result["source_documents"]
                for document in source_documents:
                    st.markdown(document.metadata['source'], help=document.page_content)
    # 응답 결과 저장
    st.session_state.messages.append({"role": "assistant", "content": response})


def user_chat(query: str):
    # 유저가 입력한 채팅을 채팅에 저장
    st.session_state.messages.append({"role": "user", "content": query})
    # 유저가 입력한 채팅을 화면에 표출
    with st.chat_message("user"):
        st.markdown(query)

# history = StreamlitChatMessageHistory(key="chat_message")
# 사용자 질문 수신 부 제작 및 출력
if query := st.chat_input("질문을 입력해주세요."):
    user_chat(query)
    assistant_chat(query)


