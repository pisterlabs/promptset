from langchain.agents import AgentExecutor, AgentType, Tool, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.prompts.chat import MessagesPlaceholder
from langchain.schema import SystemMessage

from src.config.config import OPEN_AI_LLM_MODEL, OPENAI_API_KEY
from src.tools.perfume_search_tool import PERFUME_TOOL_DESCRIPTION, PerfumeSearchTool

msgs = StreamlitChatMessageHistory(key="chat-app")


def setup_agents() -> AgentExecutor:
    llm = ChatOpenAI(
        temperature=0.0, model=OPEN_AI_LLM_MODEL, openai_api_key=OPENAI_API_KEY
    )

    perfume_search_tool = PerfumeSearchTool()

    tools = [
        Tool(
            name="perfume_search",
            func=perfume_search_tool.run,
            description=PERFUME_TOOL_DESCRIPTION,
        ),
    ]

    system_message_content = f"""You are a helpful client support agent. You have a tool at your \
disposal to help you find perfume by name, brand, or fragrance notes and filter by price. Invoke the tool with a single string query. 
You can assume that customer is asking you a question about perfume or providing more details about desired perfume.

You should greet a customer as follows: 'Hello, my name is <agent_name>. Do you want to find some perfume?'
You should respond to the customer in the same language that they are using. 
If they are using English, you should respond in English. If they are using Russian, you should respond in Russian.
Ignore product names and brand names. You should not use them to detect a language.

If user is asking for similar perfume - use provided memory for the conversation. You can use it for context about discussion.

After a customer provided you their wishes about perfume you should use available tools to find a perfume that \
suits their needs. You should search for a perfume using a tool then provide a customer with a link to the perfume that you found.
Use only results returned by the tool. You should not use any other information to find a perfume.

If user asks any question not related to perfume you should respond with 'Sorry, I can only help you with perfume \
related questions. Please ask me about perfume.'.
"""

    system_message = SystemMessage(content=system_message_content)
    agent_kwargs, memory = setup_memory_and_prompt(system_message)

    return initialize_agent(
        tools,
        llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        system_message=system_message,
        agent_kwargs=agent_kwargs,
        memory=memory,
    )


def setup_memory_and_prompt(
    system_message: SystemMessage,
) -> tuple[dict, ConversationBufferMemory]:
    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
        "system_message": system_message,
    }
    memory = ConversationBufferMemory(
        memory_key="memory", return_messages=True, chat_memory=msgs
    )

    return agent_kwargs, memory
