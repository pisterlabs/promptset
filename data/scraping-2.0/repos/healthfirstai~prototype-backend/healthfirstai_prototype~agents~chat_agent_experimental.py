"""Experimental Chat Agents

This module contains experimental chat agents that are not yet ready for production.

"""
from langchain.memory import (
    RedisChatMessageHistory,
    ConversationTokenBufferMemory,
)
from langchain.prompts import MessagesPlaceholder
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents import create_json_agent
from langchain.agents.agent_toolkits import JsonToolkit
from langchain.llms.openai import OpenAI
from langchain.tools.json.tool import JsonSpec
from langchain.agents.agent_toolkits.json.prompt import JSON_PREFIX, JSON_SUFFIX
import json
from langchain.experimental.plan_and_execute import (
    PlanAndExecute,
    load_agent_executor,
    load_chat_planner,
)

from .toolkits.user_info.tools import GetUserInfoTool
from .toolkits.diet_plan.tools import DietPlanTool, EditDietPlanTool
from .prompts import SYSTEM_PROMPT
from healthfirstai_prototype.enums.openai_enums import ModelName
from healthfirstai_prototype.utils import get_model


def init_plan_and_execute_diet_agent():
    """
    Return a PlanAndExecute object for editing a user's diet plan
    """
    planner = load_chat_planner(
        llm=get_model(ModelName.gpt_3_5_turbo),
        system_prompt=SYSTEM_PROMPT,
    )

    executor = load_agent_executor(
        llm=get_model(ModelName.gpt_3_5_turbo_0613),
        tools=[
            GetUserInfoTool(),
            DietPlanTool(),
            EditDietPlanTool(),
        ],
        verbose=True,
        include_task_in_prompt=True,
    )

    return PlanAndExecute(
        planner=planner,
        executor=executor,
        verbose=True,
    )


def start_nutrition_temp_agent(json_string):
    json_dict = json.loads(json_string)[0]
    json_spec = JsonSpec(dict_=json_dict, max_value_length=4000)
    json_toolkit = JsonToolkit(spec=json_spec)

    return create_json_agent(
        llm=OpenAI(
            client=None,
            model="text-davinci-003",
            temperature=0,
        ),
        prefix=JSON_PREFIX,
        suffix=JSON_SUFFIX,
        toolkit=json_toolkit,
        verbose=True,
    )


def init_new_agent(user_input: str, session_id="other-session", user_id: int = 1):
    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    }
    message_history = RedisChatMessageHistory(session_id=session_id)
    memory = ConversationTokenBufferMemory(
        llm=get_model(ModelName.gpt_3_5_turbo_0613),
        memory_key="memory",
        chat_memory=message_history,
        max_token_limit=2000,
        return_messages=True,
    )

    llm = get_model(ModelName.gpt_3_5_turbo_0613)
    tools = [
        DietPlanTool(),
        GetUserInfoTool(),
        EditDietPlanTool(),
    ]
    return initialize_agent(
        tools,
        llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        agent_kwargs=agent_kwargs,
        memory=memory,
    )
