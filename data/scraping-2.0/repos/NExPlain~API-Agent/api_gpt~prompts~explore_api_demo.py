import random
import webbrowser
from time import sleep
from typing import List

import chainlit as cl
from google.protobuf.json_format import MessageToJson, Parse, ParseDict
from langchain.agents import agent

from api_gpt.agents.api_agent import ApiAgent
from api_gpt.base import init_firebase_handlers
from api_gpt.data_structures.proto.generated.intent_data_pb2 import IntentData
from api_gpt.data_structures.proto.generated.workflow_pb2 import WorkflowData
from api_gpt.prompts.explore_api_prompt import (
    generate_api_exploration_workflow,
    get_api_exploration_chain,
)
from api_gpt.workflows.execute.execute import execute_intent

try:
    init_firebase_handlers(testing_environment=False)
except Exception as exception:
    print(
        f"Failed to init firebase with real data: {exception}, fallback to use testing environment. This error can be ignored if you are testing out the app."
    )
    init_firebase_handlers(testing_environment=True)

api_agent = ApiAgent()


@cl.on_chat_start
async def start():
    await display_start_message()


def format_api_info(intent_data: IntentData) -> str:
    ret = ""
    ret += f"## {intent_data.name}\n"
    ret += f"API: {intent_data.api_url}\n"
    ret += f'Inputs: {",".join(f"{input.parameter.emoji} {input.parameter.name}" for input in intent_data.inputs)}\n'
    ret += f'Outputs: {",".join(f"{output.parameter.emoji} {output.parameter.name}" for output in intent_data.outputs)}\n'
    return ret


def generate_markdown_table(fields, values, num_indents=0):
    # Calculate the maximum length of field names and values
    max_field_length = max(len(field) for field in fields)
    max_value_length = max(len(str(value)) for value in values)

    # Generate the table header
    table = f"| {'Field':<{max_field_length}} | {'Value':<{max_value_length}} |\n"
    table += f"|{'-' * (max_field_length + 2)}|{'-' * (max_value_length + 2)}|\n"

    # Generate table rows
    for field, value in zip(fields, values):
        if field == "Message":
            value = f"```\n{value}\n```"
        table += f"| {field:<{max_field_length}} | {value:<{max_value_length}} |\n"
    print(f"table : {table}")
    return table


@cl.action_callback("Start a pull request")
async def on_action(action):
    webbrowser.open("https://github.com/NExPlain/API-Agent/pulls")


def add_execute_button(intent_data: IntentData, actions: List[cl.Action]):
    """
    Adds an execute button to the actions list for a given intent.

    Args:
        intent_data (IntentData): The intent data containing information about the intent.
        actions (List[Action]): The list of actions to append the execute button to.

    Returns:
        None
    """

    execute_name = f"🚀 Run {intent_data.name}"

    @cl.action_callback(execute_name)
    async def on_action(action):
        await cl.Message(content=f"✨ Start executing {action.name}").send()
        intent_data = Parse(action.value, IntentData())

        execution_result = (await execute_intent(intent_data)).result
        if execution_result.error_code == 204:  # Need integration
            await cl.Message(
                content=f"🚧 Integration needed for executing **{intent_data.app_name}**.\n we're working on release the auto-integration using code generation, pull requests are also welcomed.",
                actions=[
                    cl.Action(
                        name="Start a pull request",
                        value="Start a pull request",
                        description="Start a pull request",
                    )
                ],
            ).send()
        elif execution_result.is_success:
            new_actions = []
            if execution_result.display_name:

                @cl.action_callback(execution_result.display_name)
                async def on_action(action):
                    if action.value:
                        webbrowser.open(action.value)

                new_actions.append(
                    cl.Action(
                        name=execution_result.display_name,
                        value=execution_result.display_link,
                        description=execution_result.display_name,
                    )
                )
            await cl.Message(
                content=f"✅ Successfully {action.name}!", actions=new_actions
            ).send()
            await action.remove()
        else:
            await cl.Message(
                content=f"Faild, {action.name} error with {execution_result.error_message}"
            ).send()

    actions.append(
        cl.Action(
            name=execute_name,
            value=MessageToJson(intent_data),
            description=f"Click to connect to {intent_data.app_name}",
        )
    )


START_MESSAGE1 = "Send a email to Zhen Li (zhen.li@plasma-ai.com)"
START_MESSAGE2 = "Book a meeting with Zhen (zhen.li@plasma-ai.com) tomorrow at 5 pm"
START_MESSAGE = f"""
# Welcome to API Agent! 🚀🚀🚀🚀🤖

Hi there, Developer! 👋 We're excited to have you on board.

## Launched products 🚀
- [Debrief AI](https://www.debrief-ai.com): The AI Notification Center that get your follow up work done automatically.
- [Plasma AI](https://www.plasma-ai.com): A platform that enables user to talk to APIs.

## Quickstart 🔗

Here are some sentences you can try to input:
- **Send email:** {START_MESSAGE1}
- **Book meeting:** {START_MESSAGE2}

### Click on example button to start:
"""


async def display_start_message():
    content = START_MESSAGE
    actions = []
    messages_names = ["Send email", "Book meeting"]
    messages_values = [START_MESSAGE1, START_MESSAGE2]
    for name, message in zip(messages_names, messages_values):
        execute_name = name

        @cl.action_callback(execute_name)
        async def on_action(action):
            await explore_workflow(user_prompt=action.value)

        actions.append(
            cl.Action(
                name=execute_name,
                value=message,
                description="Try out the example!",
            )
        )
    await cl.Message(content=content, actions=actions).send()


async def display_workflow_data(workflow_data: WorkflowData):
    content = ""
    elements = []
    actions = []

    for intent_data in workflow_data.intent_data:
        if intent_data.meta_data.logo_url:
            elements.append(
                cl.Image(
                    url=intent_data.meta_data.logo_url,
                    name=intent_data.app_name,
                    display="inline",
                    size="small",
                )
            )
        elements.append(
            cl.Text(
                content=format_api_info(intent_data),
                name=intent_data.api_url,
                display="side",
            )
        )

    content += f"Here is a workflow ready to execute:\n\n"
    content += f"### {workflow_data.name}\n\n"
    for intent_data in workflow_data.intent_data:
        content += f"- {intent_data.name}\n"
        content += f"  - App: {intent_data.app_name}\n"
        content += f"  - API: {intent_data.api_url}\n\n"
        content += f"  - Inputs\n"
        content += "\n"
        max_key_length = (
            0
            if len(intent_data.inputs) == 0
            else max(
                len(f"{input.parameter.emoji} {input.parameter.name}")
                for input in intent_data.inputs
            )
        )

        for input in intent_data.inputs:
            key_str = input.parameter.emoji + " " + input.parameter.name
            key_str += "&nbsp;" * (max_key_length - len(key_str))
            if input.parameter.type == "long_string":
                content += f"    - {key_str}:\n"
                content += f"     ```\n"
                lines = input.parameter.value.split("\n")
                for line in lines:
                    content += f"      {line}\n"
                content += "      ```\n"
            else:
                content += f"    - {key_str}: {'&nbsp;' * 3}{input.parameter.value}\n"
        content += "\n"

        add_execute_button(intent_data, actions)

    await cl.Message(content=content, elements=elements, actions=actions).send()


async def explore_workflow(user_prompt: str):
    await cl.Message(
        author="Debrief", content=f"Start generating APIs...", indent=1
    ).send()
    llm_chain = get_api_exploration_chain(mode="conversational")
    api_agent.add_user_message(user_prompt)
    workflow_data = generate_api_exploration_workflow(
        llm_chain,
        user_prompt=user_prompt,
        workflow_name=user_prompt,
        chat_history=api_agent.get_conversational_history(),
        mode="conversational",
    )
    if workflow_data is None:
        await cl.Message(
            content="Error in generating workflow, please create a issue in github with the prompt you are using. Thanks!"
        ).send()
        return
    if len(workflow_data.intent_data) > 0:
        await display_workflow_data(workflow_data)
        api_agent.clear_memory()
    else:
        raw_output = workflow_data.name
        api_agent.add_ai_message(raw_output)
        await cl.Message(
            content=api_agent.format_ai_message_for_display(raw_output)
        ).send()


@cl.on_message  # this function will be called every time a user inputs a message in the UI
async def main(message: str):
    await explore_workflow(message)
