import streamlit as st
from streamlit_chat import message
import openai


st.info("vicuna-13b模型测试")


def openai_create(messages):
        openai.api_key = "EMPTY"  # Not support yet
        openai.api_base = "http://localhost:38080/v1"

        model = "vicuna-13b"

        # create a chat completion
        completion = openai.ChatCompletion.create(
            model=model,
            messages=messages
        )
        # print the completion
        print(completion.choices[0].message.content)
        return completion.choices[0].message.content


def conversation():
    # text_input = {"中文": "输入后按回车键发送消息(清空上下文请输入clear)"}

    if 'generated' not in st.session_state:
        st.session_state['generated'] = []

    if 'past' not in st.session_state:
        st.session_state['past'] = []

    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
        # st.session_state.messages.append(conversation_init)


    def get_text():
        input_text = st.text_input("输入后按回车键发送消息(清空上下文请输入clear)", key="role_conversation_input")
        return input_text

    question = get_text()
    if question:
        if question == "clear":
            st.session_state.generated = []
            st.session_state.past = []
            st.session_state.messages = []
            # st.session_state.messages.append(conversation_init)
        else:
            conversation = {}
            conversation["role"] = "user"
            conversation["content"] = question
            st.session_state.messages.append(conversation)
            prompt = st.session_state.messages
            result = openai_create(prompt)
            st.session_state.generated.append(result)
            st.session_state.past.append(question)
            conversation = {}
            conversation["role"] = "assistant"
            conversation["content"] = result
            st.session_state.messages.append(conversation)
            print(st.session_state.messages)
            if len(st.session_state.messages) > 10:
                del st.session_state.messages[9:-1]

    if st.session_state['generated'] and st.session_state['past']:
        for i in range(len(st.session_state['past']) - 1, -1, -1):
            try:
                message(st.session_state["generated"][i], key=str(i))
                message(st.session_state['past'][i],
                    is_user=True, key=str(i) + '_user')
            except Exception as exc:
                print(exc)
                st.error("网络出错啦，请刷新或者输入clear清除上下文再试试吧~")

def formatted():
    st.title("Vicuna-13B格式保持")
    input_words = st.text_area("请输入问题:", key="question_input")

    prompt = [{"role": "user", "content": input_words}]

    max_input_len = 2000

    if st.button("确认", key="word_gpt3"):
                if input_words.__len__() < max_input_len:
                    with st.spinner('答案生成中...'):
                        result = openai_create(prompt)
                        st.balloons()
                        st.success("大功告成！")
                        st.markdown(result)

tab1, tab2= st.tabs(["💻Vicuna-13B格式保持","💁‍与Vicuna-13B对话"])

with tab1:
    formatted()

with tab2:
    conversation()
