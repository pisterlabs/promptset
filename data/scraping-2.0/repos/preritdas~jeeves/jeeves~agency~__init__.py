"""Agent GPT."""
from langchain.agents import Tool, ZeroShotAgent, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import OutputParserException

import uuid
import pytz
import datetime as dt

from keys import KEYS
from config import CONFIG
from jeeves.permissions import User

from jeeves.agency import tool_auth
from jeeves.agency.chat_history.models import Message
from jeeves.agency import logs_callback, prompts
from jeeves.agency.chat_history import ChatHistory


# ---- Build the agent ----

class InternalThoughtZeroShotAgent(ZeroShotAgent):
    """
    A normal ZeroShotAgent but doesn't inject "Thought:" before the LLM. After testing
    and heavy prompt engineering, I've found a better sucess rate with having the LLM
    create its own "Thought" label. This is because it knows that each Thought must
    also have either an Action/Action Input or a Final Answer.
    """
    @property
    def llm_prefix(self) -> str:
        """Prefix to append the llm call with."""
        return ""


# --- Create the LLM and AgentExecutor ---

llm = ChatOpenAI(
    model_name=CONFIG.GPT.base_openai_model, 
    openai_api_key=KEYS.OpenAI.api_key, 
    temperature=CONFIG.GPT.temperature
)


def create_agent_executor(
    toolkit: list[Tool],
    user: User,
    callback_handlers: list[BaseCallbackHandler],
) -> AgentExecutor:
    """Create the agent given authenticated tools."""
    agent_prompts: prompts.AgentPrompts = prompts.build_prompts(user)
    agent = InternalThoughtZeroShotAgent.from_llm_and_tools(
        llm=llm,
        tools=toolkit,
        handle_parsing_errors=True,
        prefix=agent_prompts.prefix,
        format_instructions=agent_prompts.format_instructions,
        suffix=agent_prompts.suffix
    )
    return AgentExecutor(
        agent=agent,
        tools=toolkit,
        max_iterations=50,
        verbose=True,
        callbacks=callback_handlers
    )


def create_base_agent_executor(
    toolkit: list[Tool],
    callback_handlers: list[BaseCallbackHandler],
) -> AgentExecutor:
    """Create the agent executor without a User object."""
    agent_prompts: prompts.AgentPrompts = prompts.build_base_agent_prompts()
    agent = InternalThoughtZeroShotAgent.from_llm_and_tools(
        llm=llm,
        tools=toolkit,
        handle_parsing_errors=True,
        prefix=agent_prompts.prefix,
        format_instructions=agent_prompts.format_instructions,
        suffix=agent_prompts.suffix
    )
    return AgentExecutor(
        agent=agent,
        tools=toolkit,
        max_iterations=50,
        verbose=True,
        callbacks=callback_handlers
    )


# ---- Run the agent ----

def retry_couldnt_parse(function):
    """Decorator to retry up to three times if a specific ValueError occurs."""
    def wrapper(*args, **kwargs):
        retries = 0
        last_exception = None
        while retries < 3:
            try:
                return function(*args, **kwargs)
            except OutputParserException as e:
                if "Could not parse LLM output" in str(e):
                    retries += 1
                    last_exception = e
                else:
                    raise e
        raise last_exception

    return wrapper


@retry_couldnt_parse
def run_agent(agent_executor: AgentExecutor, query: str, uid: str) -> str:
    """Run the agent."""
    with get_openai_callback() as cb:
        res = agent_executor.run(query)
        logs_callback.logger.info(
            f"{uid}: UsageInfo: "
            f"Total Tokens: {cb.total_tokens}, "
            f"Prompt Tokens: {cb.prompt_tokens}, "
            f"Completion Tokens: {cb.completion_tokens}, "
            f"Total Cost (USD): ${cb.total_cost:.2f}."
        )
        return res


def _create_uid() -> str:
    """Create a unique ID for an agent run."""
    return str(uuid.uuid4())


def generate_agent_response(content: str, user: User, uid: str = "") -> str:
    """Build tools, create executor, and run the agent. UID is optional."""
    uid = uid or _create_uid()
    assert user

    # Build chat history and toolkit using inbound phone
    ChatHistory.from_inbound_phone(user.phone)
    callback_handlers = logs_callback.create_callback_handlers(uid)
    toolkit = tool_auth.build_tools(user, callback_handlers)

    # Run
    agent_executor = create_agent_executor(
        toolkit, user, callback_handlers
    )
    response: str = run_agent(agent_executor, content, uid)

    # Save message to chats database
    ChatHistory.from_inbound_phone(user.phone).add_message(
        Message(
            datetime=dt.datetime.now(pytz.timezone(CONFIG.General.default_timezone)),
            inbound_phone=user.phone,
            user_input=content,
            agent_response=response
        )
    )

    return response.strip()


def generate_base_agent_response(content: str, uid: str = "") -> str:
    """Create executor and run the agent. UID is optional."""
    # Use overridden uid or create a new one
    uid = uid or _create_uid()

    # Build toolkit using default callback handlers
    callback_handlers = logs_callback.create_callback_handlers(uid)
    toolkit = tool_auth.NO_AUTH_TOOLS

    # Insert callback handlers for all tools
    for tool in toolkit:
        tool.callbacks = callback_handlers

    # Run
    agent_executor = create_base_agent_executor(toolkit, callback_handlers)
    response: str = run_agent(agent_executor, content, uid)

    return response.strip()
