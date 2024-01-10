import re
from datetime import timedelta

from langchain import OpenAI
from langchain.agents import initialize_agent, AgentType, load_tools
from langchain.chat_models import ChatOpenAI
from langchain.schema import OutputParserException

from .summarize_chat_tool import SummarizeChatHistory
from .summarize_text_tool import SummarizeText
from .util_tools import say, today, date_difference, time_since_today
from ...events import Context, CommandResult, msg_cmd, Message, message_handler


@message_handler
def handle_message(ctx: Context, msg: Message) -> CommandResult:
    if msg.sent_before(timedelta(minutes=30)):
        return

    if not msg.mentioned_me:
        return

    txt = msg.text_replace_my_mentions("")

    # cmd = '!'
    # if not txt.startswith(cmd):
    #     return
    #
    # txt = txt[len(cmd):].strip()
    # if txt == "":
    #     return

    llm1 = OpenAI(temperature=0)
    llm = ChatOpenAI(temperature=0)

    tools = load_tools(["requests_all", "llm-math", "open-meteo-api", "wikipedia"], llm=llm1)
    tools += [
        SummarizeChatHistory(llm=llm, store=ctx.store, chat_jid=msg.chat, my_jid=msg.my_jid),
        say,
        time_since_today,
        today,
        date_difference,
        SummarizeText(llm=llm),
    ]

    prefix = """Bot is a group chat bot that uses large language model trained by OpenAI.
Bot is designed to help the chat members to navigate the chat conversation, as well as to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.
Answer the following questions as best you can. You have access to the following tools:"""
    agent = initialize_agent(tools, llm1, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, prefix=prefix)

    try:
        response = agent.run(txt)
    except OutputParserException as e:
        response = str(e)

        pattern = r'say\("([^"]*)"\)'
        match = re.search(pattern, response)
        if match:
            response = match.group(1)
    yield msg_cmd(msg.chat, f"@{msg.sender_jid.user}, {response}")
