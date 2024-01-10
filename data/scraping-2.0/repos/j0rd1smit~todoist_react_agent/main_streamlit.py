import json
import os
import time

import openai
import pydantic
import streamlit as st
import tqdm
from dotenv import load_dotenv
from streamlit_chat import message

from todoist_react_agent.chat_bot import ChatBot
from todoist_react_agent.models import (
    CreateNewProjectAction,
    GetAllInboxTasksAction,
    GetAllProjectsAction,
    GetAllTasksAction,
    GiveFinalAnswerAction,
    MoveTaskAction,
    ReactResponse,
)
from todoist_react_agent.repair_agent import parse_base_model_with_retries
from todoist_react_agent.todoist_action_toolkit import TodoistActionToolKit


def main() -> None:
    st.set_page_config(page_title="ToDo Agent", page_icon=":robot:")

    st.header("ToDo Agent")

    # max tokens options to mix from [128, 256, 512, 1024]

    st.sidebar.header("Model settings")
    max_tokens = st.sidebar.select_slider(
        "Max response tokens", [128, 256, 512, 1024], value=512
    )

    # selection box for the engine:
    history_length = st.sidebar.slider("Message History length:", 1, 25, 15)
    temperature = st.sidebar.slider("Temperature:", 0.0, 1.0, 0.0, 0.1)
    max_actions = st.sidebar.slider("Max number of actions:", 1, 50, 20)
    st.sidebar.header("Instructions")
    user_input = st.sidebar.text_area(
        "What do you want the agent to do?",
        placeholder="Which tasks do I have in my inbox?",
    )
    submit_button = st.sidebar.button("Submit")

    if submit_button:
        render_agent_loop(
            user_input,
            history_length=history_length,
            temperature=temperature,
            max_actions=max_actions,
            max_tokens=max_tokens,
        )

    else:
        st.write("Please specify what you want the agent to do in the sidebar.")


def render_agent_loop(
    user_input: str,
    history_length: int,
    temperature: float,
    max_actions: int,
    max_tokens: int,
) -> None:
    out_of_order_render = OutOfOrderRender()
    message_render = MessageRender()

    with out_of_order_render:
        message_render(user_input, is_user=True)

    system_message = create_system_prompt(ReactResponse, user_input)
    chatbot = ChatBot(
        system_message=system_message,
        history_length=history_length,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    todoist = TodoistActionToolKit(os.getenv("TODOIST_API_KEY"))

    inputs = json.dumps({"objective": user_input})
    for i in range(max_actions):
        raw_response = chatbot(inputs, role="user")
        try:
            response = parse_base_model_with_retries(raw_response, ReactResponse)
            with out_of_order_render:
                message_render(
                    f"Thought: {response.thought}\n\nAction: {response.action.dict()}\n\nNumber of actions used: {i + 1}"
                )

            chatbot.set_message_content(-1, json.dumps(response.dict()))

            match response.action:
                case GiveFinalAnswerAction():
                    with out_of_order_render:
                        message_render(f"Final Answer: {response.action.answer}")
                    return
                case GetAllInboxTasksAction():
                    observation = todoist.get_inbox_tasks()
                case GetAllTasksAction():
                    observation = todoist.get_all_tasks()
                case GetAllProjectsAction():
                    observation = todoist.get_all_projects()
                case MoveTaskAction(task_id=task_id, project_id=project_id):
                    todoist.move_task(task_id, project_id)
                    observation = (
                        f"Task with id {task_id} moved to project with id {project_id}."
                    )
                case CreateNewProjectAction(project_name=project_name):
                    observation = todoist.create_project(project_name)
                case _:
                    raise ValueError(f"Unknown action {response.action}")
        except ValueError as e:
            observation = f"You response caused the following error: {e}. Please try again and avoid this error."

            chatbot.set_message_content(-1, json.dumps(observation))

        with out_of_order_render:
            message_render(f"Observation: {observation}")
        inputs = json.dumps({"observation": observation})

    with out_of_order_render:
        message_render(f"I have used my maximum number of actions. I will now stop.")


def create_system_prompt(react_model: pydantic.BaseModel, question: str) -> str:
    return f"""
You are a getting things done (GTD) agent.
It is your job to accomplish the following task: {question}
You have access to multiple tools to accomplish this task.
See the action in the json schema for the available tools.
If you have insufficient information to answer the question, you can use the tools to get more information.
All your answers must be in json format and follow the following schema json schema:
{react_model.schema()}

If your json response asks me to preform an action, I will preform that action.
I will then response with the result of that action.

Let's begin to answer the question: {question}
Do not write anything else than json!
"""


class MessageRender:
    def __init__(self):
        self.idx = 0

    def __call__(self, message_str: str, is_user: bool = False):
        message(message_str, is_user=is_user, key=f"message_{self.idx}")
        self.idx += 1


class OutOfOrderRender:
    def __init__(self, max_n_elements: int = 200):
        self.max_n_elements = max_n_elements
        self.placeholders = [st.empty() for _ in range(max_n_elements)]
        self.placeholder_idx = max_n_elements - 1

    def __enter__(self):
        self.placeholders[self.placeholder_idx].__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.placeholders[self.placeholder_idx].__exit__(exc_type, exc_val, exc_tb)
        self.placeholder_idx = self.placeholder_idx - 1


if __name__ == "__main__":
    load_dotenv()
    openai.api_type = "azure"
    openai.api_version = "2023-03-15-preview"
    openai.api_base = os.environ["OPENAI_API_BASE"]
    openai.api_key = os.environ["OPENAI_API_KEY"]

    main()
