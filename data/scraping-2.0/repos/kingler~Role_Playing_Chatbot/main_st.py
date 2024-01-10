import streamlit as st
import os
from termcolor import colored
from typing import List
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)
from camel_agent import CAMELAgent
from inception_prompts import assistant_inception_prompt, user_inception_prompt
import json


# Function to select roles
def select_role(role_type, roles):
    selected_role = st.selectbox(f"Select {role_type} role:",  ["Custom Role"] + roles )
    if selected_role == "Custom Role":
        custom_role = st.text_input(f"Enter the {role_type} (Custom Role):")
        return custom_role
    else:
        return selected_role

# Function to get system messages for AI assistant and AI user from role names and the task
def get_sys_msgs(assistant_role_name: str, user_role_name: str, task: str):
    assistant_sys_template = SystemMessagePromptTemplate.from_template(template=assistant_inception_prompt)
    assistant_sys_msg = assistant_sys_template.format_messages(assistant_role_name=assistant_role_name, user_role_name=user_role_name, task=task)[0]

    user_sys_template = SystemMessagePromptTemplate.from_template(template=user_inception_prompt)
    user_sys_msg = user_sys_template.format_messages(assistant_role_name=assistant_role_name, user_role_name=user_role_name, task=task)[0]

    return assistant_sys_msg, user_sys_msg

def generate_unique_task_name(task: str, chat_history_items: List[dict]) -> str:
    task_name = task
    count = 1
    task_names = [item["task"] for item in chat_history_items]

    while task_name in task_names:
        task_name = f"{task} ({count})"
        count += 1

    return task_name

def load_chat_history_items() -> List[dict]:
    chat_history_items = []
    try:
        with open("chat_history.json", "r") as history_file:
            for line in history_file:
                chat_history_items.append(json.loads(line.strip()))
    except FileNotFoundError:
        pass

    return chat_history_items


chat_history_items = []


st.set_page_config(layout="centered") 

st.title("OmniSolver 🔆", help="This app uses the CAMEL framework to solve problems. This app uses GPT models and the responses may not be accurate")

# Sidebar: API Key input
st.sidebar.title("Configuration")
# comment this out if you want to use the API key from the environment variable locally
api_key = st.sidebar.text_input("Enter your OpenAI API Key:", type="password")

# uncomment this if you want to use the API key from the environment variable locally
# api_key = os.getenv("OPENAI_API_KEY")

if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
elif api_key == "":
    st.sidebar.warning("Please enter your OpenAI API Key.")

  

# Sidebar: Model selection
model = st.sidebar.radio("Select the model:", ("gpt-3.5-turbo", "gpt-4"))

with open("stats.txt", "r") as stats_file:
                    stats = stats_file.readlines()
                    tasks_solved = int(stats[0].strip())
                    tasks_solved += 1
                    st.write(f"<p style='color: green; font-weight: bold;'>This App was used to solve *{tasks_solved}* tasks so far since deployed</p>", unsafe_allow_html=True)
# Main: Load roles from roles.txt
with open("roles.txt", "r") as roles_file:
    roles_list = [line.strip() for line in roles_file.readlines()]

# Main: Role selection
user_role_name = select_role("AI user", roles_list)
assistant_role_name = select_role("AI assistant", roles_list)


if assistant_role_name and user_role_name:
    # Main: Task input
    task = st.text_input("Please enter the task:")

    if task:
        # Main: Task specifier
        task_specifier = st.checkbox("Do you want to use the task specifier feature?", help="Use the task specifier feature to make a task more specific by GPT. May not work as expected.")

        if task_specifier:
            word_limit = st.number_input("Please enter the word limit for the specified task:", min_value=1, value=50, step=1)

            if word_limit:
                task_specifier_sys_msg = SystemMessage(content="You can make a task more specific.")
                task_specifier_prompt = (
                    """Here is a task that {assistant_role_name} will help {user_role_name} to complete: {task}.
                    Please make it more specific. Be creative and imaginative.
                    Please reply with the specified task in {word_limit} words or less. Do not add anything else."""
                )
                task_specifier_template = HumanMessagePromptTemplate.from_template(template=task_specifier_prompt)
                task_specify_agent = CAMELAgent(task_specifier_sys_msg, ChatOpenAI(model=model, temperature=1.0))
                task_specifier_msg = task_specifier_template.format_messages(assistant_role_name=assistant_role_name,
                                                            user_role_name=user_role_name,
                                                            task=task, word_limit=word_limit)[0]
                specified_task_msg = task_specify_agent.step(task_specifier_msg)
                st.write(f"<p style='font-weight: bold;'>Specified task:</p> {specified_task_msg.content}", unsafe_allow_html=True)

                specified_task = specified_task_msg.content
        else:
            specified_task = task

        if specified_task:
            # Main: Chat turn limit input
            chat_turn_limit = st.number_input("Please enter the chat turn limit:", min_value=1, step=1)
            
            if st.button("Start Solving Task"):
                if api_key == "":
                    st.warning("Please enter your OpenAI API Key.")
                    st.stop()

                with open("stats.txt", "w") as stats_file:
                    stats_file.write(str(tasks_solved))

                chat_history_items = load_chat_history_items()
                with st.spinner("Thinking..."):
                    # Main: Initialize agents and start role-playing session
                    assistant_sys_msg, user_sys_msg = get_sys_msgs(assistant_role_name, user_role_name, specified_task)
                    assistant_agent = CAMELAgent(assistant_sys_msg, ChatOpenAI(model=model, temperature=0.2))
                    user_agent = CAMELAgent(user_sys_msg, ChatOpenAI(model=model, temperature=0.2))

                    assistant_agent.reset()
                    user_agent.reset()

                    assistant_msg = HumanMessage(
                        content=(f"{user_sys_msg.content}. "
                                "Now start to give me introductions one by one. "
                                "Only reply with Instruction and Input."))

                    user_msg = HumanMessage(content=f"{assistant_sys_msg.content}")
                    user_msg = assistant_agent.step(user_msg)

                    st.write(f"<p style='color: red;'><b>Original task prompt:</b></p>\n\n{task}\n", unsafe_allow_html=True)
                    st.write(f"<p style='color: red;'><b>Specified task prompt:</b></p>\n\n{specified_task}\n", unsafe_allow_html=True)


                    chat_history = []

                    with st.spinner("Running role-playing session to solve the task..."):
                        # Replace the for loop with the following code:
                        progress = st.progress(0)
                        for n in range(chat_turn_limit):
                            user_ai_msg = user_agent.step(assistant_msg)
                            user_msg = HumanMessage(content=user_ai_msg.content)

                            chat_history.append({"role": user_role_name, "content": user_msg.content})
                            st.markdown(f"<p style='color: blue; font-weight: bold;'>{user_role_name}</p>\n\n{user_msg.content}\n\n", unsafe_allow_html=True)

                            assistant_ai_msg = assistant_agent.step(user_msg)
                            assistant_msg = HumanMessage(content=assistant_ai_msg.content)

                            chat_history.append({"role": assistant_role_name, "content": assistant_msg.content})
                            st.markdown(f"<p style='color: green; font-weight: bold;'>{assistant_role_name}</p>\n\n{assistant_msg.content}\n\n", unsafe_allow_html=True)

                            progress.progress((n+1)/chat_turn_limit)

                            if "<CAMEL_TASK_DONE>" in user_msg.content:
                                break

                        progress.empty()


                    

                    # Main: Save chat history to file
                    task_name = generate_unique_task_name(task, chat_history_items)
                    history_dict = {
                        "task": task_name,
                        "settings": {
                            "assistant_role_name": assistant_role_name,
                            "user_role_name": user_role_name,
                            "model": model,
                            "chat_turn_limit": chat_turn_limit,
                        },
                        "conversation": chat_history,
                    }

                    with open("chat_history.json", "a") as history_file:
                        json.dump(history_dict, history_file)
                        history_file.write("\n")
                    

            else:
                st.warning("Please enter the chat turn limit.")
        else:
            st.warning("Please specify the task.")
    else:
        st.warning("Please enter the task.")
else:
    st.warning("Please select both AI assistant and AI user roles.")

# Sidebar: Load chat history
chat_history_titles = [item["task"] for item in chat_history_items]
try:
    chat_history_items = load_chat_history_items()
    chat_history_titles = [item["task"] for item in chat_history_items]
    selected_history = st.sidebar.selectbox("Select chat history:", ["None"] + chat_history_titles)

    if selected_history != "None":
        delete_history_button = st.sidebar.button("Delete Selected Chat History")

        if delete_history_button and selected_history != "None":
            chat_history_items.pop(chat_history_titles.index(selected_history))

            # Save the updated chat history to file
            with open("chat_history.json", "w") as history_file:
                for item in chat_history_items:
                    json.dump(item, history_file)
                    history_file.write("\n")

            st.sidebar.success("Selected chat history deleted.")
            st.experimental_rerun()



    # Main: Display selected chat history
    if selected_history != "None":
        selected_history_item = chat_history_items[chat_history_titles.index(selected_history)]
        settings = selected_history_item["settings"]
        conversation = selected_history_item["conversation"]

        st.write(f"<p style='color: green; font-weight: bold;'>Task:</p> {selected_history}\n", unsafe_allow_html=True)

        st.write(f"""<p style='color: green; font-weight: bold;'>Settings:</p>
                    <p>- AI assistant role: <span >{settings['assistant_role_name']}</span></p>
                    <p>- AI user role: <span >{settings['user_role_name']}</span></p>
                    <p>- Model: {settings['model']}</p>
                    <p>- Chat turn limit: {settings['chat_turn_limit']}</p>
                    """, unsafe_allow_html=True)


        for msg in conversation:
            st.markdown(f"<p style='color: green; font-weight: bold;'>{msg['role']}</p>\n\n{msg['content']}\n\n", unsafe_allow_html=True)

except FileNotFoundError:
    st.sidebar.warning("No chat history available.")

st.sidebar.title("Search for more projects like this at [echohive](https://www.echohive.live)")