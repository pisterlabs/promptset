import os
from collections import namedtuple
from time import sleep

import openai
import streamlit as st


st.set_page_config(page_title="AI Chat Room", page_icon="💬")

st.title("AI Chat Room")


ss = st.session_state

with st.sidebar:
    ss.openai_api_key = st.text_input("OpenAI API key", placeholder="sk-xxxx", type="password")
    ss.bot_num = st.number_input("Number of bots", value=2, min_value=2, max_value=9)

if key := (ss.openai_api_key or os.getenv("OPENAI_API_KEY")):
    oai = openai.OpenAI(api_key=key)
else:
    st.warning("Please set your OpenAI API key")

class Bot:
    def __init__(
        self,
        name: str,
        instructions: str,
        avatar: str = "🤖",
        tools: list[dict] = [],
        model: str = "gpt-3.5-turbo-1106",
    ) -> None:
        self.name = name
        self.avatar = avatar
        self.assistant = oai.beta.assistants.create(
            name=name,
            instructions=instructions,
            tools=tools,
            model=model,
        )
        self.thread = oai.beta.threads.create()

Message = namedtuple("Message", ["index", "content"])


st.info("This is an AI chat room for a group of bots, which you can configure below:")

with st.expander("Bots Configuration", expanded=True):
    with st.form("config"):
        configs = [{} for _ in range(ss.bot_num)]
        cols = st.columns(min(ss.bot_num, 3))
        while len(cols) < ss.bot_num:
            cols.extend(st.columns(min(ss.bot_num - len(cols), 3)))
        for i, col in enumerate(cols):
            with col:
                st.subheader(f"Bot {i+1}")
                col1, col2 = st.columns([3, 1])
                with col1:
                    configs[i]["name"] = st.text_input("Name", value="Alice", key=f"bot_name_{i}")
                with col2:
                    configs[i]["avatar"] = st.text_input("Avatar", value="👩", key=f"bot_avatar_{i}")
                configs[i]["instructions"] = st.text_area("Instructions", value="You are Alice. You are in a chat room.", key=f"bot_instructions_{i}")

        init_message = st.text_input(
            "Initial message",
            value="Hi! I'm Alice. How are you guys?",
            help="The initial message that will be sent from the first bot to others."
        )
        if st.form_submit_button("Save"):
            ss.bots = [
                Bot(config["name"], config["instructions"], config["avatar"])
                for config in configs
            ]


box = st.container()


if "bots" in ss:

    if "messages" not in ss:
        ss.messages = [Message(0, init_message)]
    
    for message in ss.messages:
        with box.chat_message(name=ss.bots[message.index].name, avatar=ss.bots[message.index].avatar):
            st.markdown(message.content, unsafe_allow_html=True)

    chat = [
        col.button(f"{ss.bots[i].name} Say", key=f"chat_{i}")
        for i, col in enumerate(st.columns(ss.bot_num))
    ]
    if any(chat):
        index = chat.index(True)
        bot = ss.bots[index]
        index_arr = [m.index for m in ss.messages][::-1]
        new_from = index_arr.index(index) if index in index_arr else 0
        for message in ss.messages[-new_from:]:
            _ = oai.beta.threads.messages.create(
                thread_id=bot.thread.id,
                role="user",
                content=f"<!-- {ss.bots[message.index].name} -->\n{message.content}",
            )
        run = oai.beta.threads.runs.create(
            thread_id=bot.thread.id,
            assistant_id=bot.assistant.id
        )
        while (
            status := oai.beta.threads.runs.retrieve(run.id, thread_id=bot.thread.id).status
        ) not in ("completed", "failed"):
            sleep(2)
        
        if status == "completed":
            messages = oai.beta.threads.messages.list(bot.thread.id, order="asc")
            new_messages = [msg for msg in messages if msg.run_id == run.id]
            answer = [
                msg_content for msg in new_messages for msg_content in msg.content
            ]
            if all(
                isinstance(content, openai.types.beta.threads.MessageContentText)
                for content in answer
            ):
                answer = "\n".join(content.text.value for content in answer)
        else:
            answer = f"Error: {run.last_error}"
        with box.chat_message(name=bot.name, avatar=bot.avatar):
            st.markdown(answer, unsafe_allow_html=True)
        
        ss.messages.append(Message(index, answer))
