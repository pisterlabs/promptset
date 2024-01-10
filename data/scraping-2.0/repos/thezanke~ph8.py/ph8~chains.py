from langchain.agents import Tool
from langchain.chains import OpenAIModerationChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.tools.render import format_tool_to_openai_function
from ph8.preferences import Preferences
from ph8.scraping import get_text_content_from_url
from langchain.agents import AgentExecutor
import discord
import discord.ext.commands as commands
import logging
import ph8.config

logger = logging.getLogger(__name__)

lc_tools = [
    Tool.from_function(
        func=get_text_content_from_url,
        name="WebBrowser",
        description="Useful for when you want to load the text content from a web URL",
    )
]

oai_functions = [format_tool_to_openai_function(t) for t in lc_tools]


async def ainvoke_conversation_chain(
    bot: commands.Bot,
    message: discord.Message,
    reply_chain: list[discord.Message | discord.DeletedReferencedMessage],
):
    assert bot.user is not None, "Bot user is not set"

    preferences: Preferences = bot.get_cog("Preferences")  # type: ignore
    assert preferences is not None, "Preferences cog is not loaded"

    model_name = preferences.get_user_pref(message.author.id, "model_name")

    llm = ChatOpenAI(model=model_name, temperature=0.7, max_tokens=420)
    moderation = OpenAIModerationChain(client=llm.client)

    modded_content = await moderation.arun(message.content)

    if modded_content != message.content:
        return modded_content

    llm_with_tools = llm.bind(functions=oai_functions)

    input_args = {
        "bot_id": bot.user.id,
        "bot_name": bot.user.display_name,
        "author_id": message.author.id,
        "author_name": message.author.display_name,
        "user_message": message.content,
    }

    messages = [
        ("system", ph8.config.conversation.system_message_intro),
        ("system", "CONTEXT.ASSISTANT:\n\n* NAME: {bot_name}\n* ID: {bot_id}"),
    ]

    agent_arg_map = {
        "user_message": lambda x: x["user_message"],
        "bot_id": lambda x: x["bot_id"],
        "bot_name": lambda x: x["bot_name"],
        "author_id": lambda x: x["author_id"],
        "author_name": lambda x: x["author_name"],
        "message_history": lambda x: x.get("message_history", None),
        "agent_scratchpad": lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]
        ),
    }

    if len(reply_chain):
        messages.append(("system", "CONTEXT.MESSAGE_HISTORY:\n\n{message_history}"))

        history: list[str] = []

        for m in reply_chain:
            if isinstance(m, discord.DeletedReferencedMessage):
                history.append("<deleted message>")
                continue

            history.append(f"{m.author.display_name} (ID: {m.author.id}): {m.content}")

        input_args["message_history"] = "\n".join(history)

    messages.append(
        (
            "system",
            "CONTEXT.MESSAGE_AUTHOR:\n\n* Name:{author_name}\n* ID: {author_id}",
        )
    )
    messages.append(MessagesPlaceholder(variable_name="agent_scratchpad"))
    messages.append(("user", "{user_message}"))

    prompt = ChatPromptTemplate.from_messages(messages)
    agent = agent_arg_map | prompt | llm_with_tools | OpenAIFunctionsAgentOutputParser()
    agent_executor = AgentExecutor(agent=agent, tools=lc_tools, verbose=True)  # type: ignore
    response = await agent_executor.ainvoke(input_args)

    return response["output"]
