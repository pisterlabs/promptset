import time

import predictionguard as pg
from langchain import PromptTemplate, FewShotPromptTemplate
import streamlit as st

#--------------------------#
# Prompt templates         #
#--------------------------#

demo_formatter_template = """\nUser: {user}
Assistant: {assistant}\n"""
demo_prompt = PromptTemplate(
    input_variables=["user", "assistant"],
    template=demo_formatter_template,
)


#---------------------#
# Streamlit config    #
#---------------------#

#st.set_page_config(layout="wide")

# Hide the hamburger menu
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


#--------------------------#
# Streamlit sidebar        #
#--------------------------#

st.sidebar.title("聊天游乐场")
st.sidebar.markdown(
    "这是基于 [Prediction Guard](https://www.predictionguard.com) 的聊天助手的游乐场。 "
    "您可以尝试不同的模型和配置，看看助手如何响应。"
)

st.sidebar.markdown("## 型号配置")
model = st.sidebar.selectbox(label="型号", options=["Yi-34B"])
temperature = st.sidebar.slider(
    label="温",
    min_value=0.0,
    max_value=1.0,
    value=0.75,
    step=0.01,
    format="%f",
)
max_tokens = st.sidebar.slider(
    label="最大代币数",
    min_value=1,
    max_value=1000,
    value=200,
    step=10,
    format="%d",
)

st.sidebar.markdown("## 视察/守卫")
consistency = st.sidebar.checkbox("一致", value=False)
factuality = st.sidebar.checkbox("事实性", value=False)
toxicity = st.sidebar.checkbox("毒性", value=False)


#--------------------------#
# Streamlit app            #
#--------------------------#

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("怎么了?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # contruct prompt
        examples = []
        turn = "user"
        example = {}
        for m in st.session_state.messages:
            latest_message = m["content"]
            example[turn] = m["content"]
            if turn == "user":
                turn = "assistant"
            else:
                turn = "user"
                examples.append(example)
                example = {}
        if len(example) > 4:
            examples = examples[-4:]

        few_shot_prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=demo_prompt,
            example_separator="",
            prefix="以下是人工智能助手和人类用户之间的对话。助理乐于助人、富有创意、聪明且非常友好。\n",
            suffix="\n人类: {human}\n助手: ",
            input_variables=["human"],
        )

        content = few_shot_prompt.format(human=latest_message)

        # generate response
        with st.spinner("思维..."):
            result = pg.Completion.create(
                model=model,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                output = {
                    "consistency": consistency,
                    "factuality": factuality,
                    "toxicity": toxicity
                }
            )
        if "error" in result['choices'][0]['status']:
            warning = "> ⚠️" + result['choices'][0]['status']
            message_placeholder.markdown(warning)
            full_response = warning
        else:
            completion = result['choices'][0]['text']
            completion = completion.split("Human:")[0].strip()
            completion = completion.split("H:")[0].strip()
            completion = completion.split('#')[0].strip()
            for token in completion.split(" "):
                full_response += " " + token
                message_placeholder.markdown(full_response + "▌")
                time.sleep(0.075)
            message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})