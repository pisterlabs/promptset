from typing import Any
from typing import Dict
from typing import List
from operator import itemgetter

from dotenv import load_dotenv

from langchain.agents import tool
from langchain.agents.agent import AgentExecutor
from langchain.agents.agent import AgentFinish
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import MessagesPlaceholder
from langchain.schema import StrOutputParser
from langchain.tools.render import format_tool_to_openai_function

from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableSequence

from langchain_community.chat_message_histories.in_memory import ChatMessageHistory

load_dotenv()

# llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
llm = ChatOpenAI(model="gpt-4", temperature=0)


@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)


# can add more tools here, they will be available to the agent
tools = [get_word_length]

tool_dict = {f.name: f for f in tools}

MEMORY_KEY = "chat_history"
AGENT_SCRATCHPAD_KEY = "agent_scratchpad"


def create_prompt(sys_prompt: str):
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                sys_prompt,
            ),
            MessagesPlaceholder(variable_name=MEMORY_KEY),
            ("user", "{input}"),
            # MessagesPlaceholder(variable_name=AGENT_SCRATCHPAD_KEY),
        ]
    )


llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])


class AiMessageWithID(AIMessage):
    agent_id: str


class ChatMessageHistoryWithLogging(ChatMessageHistory):
    def add_message(self, message):
        print(f"saving ({message = }) to (memory id = {id(self)})")
        super().add_message(message)
        print("------------------start")
        for m in self.messages:
            print(m)
        print("------------------end\n\n")


class MemoryWithId(ConversationSummaryBufferMemory):
    chat_memory = ChatMessageHistoryWithLogging()

    def load_memory_for_refereee(self):
        """Dumps memory to a string like

        agent 1: bla
        agent 2: blabla
        agent 1: blablabla
        agent 2: blu
        """
        adjusted_messages_by_id = []
        mssages_with_id: List[AiMessageWithID] = self.load_memory_variables({})[
            MEMORY_KEY
        ]
        for msg in mssages_with_id:
            if isinstance(msg, SystemMessage):
                adjusted_messages_by_id.append(msg.content)
            if isinstance(msg, HumanMessage):
                adjusted_messages_by_id.append("agent_1: " + msg.content)
            if isinstance(msg, AIMessage):
                adjusted_messages_by_id.append("agent_2: " + msg.content)

        return "\n".join(adjusted_messages_by_id)


def handleAgentOutput(memory: MemoryWithId):
    def _handle(output):
        if isinstance(output, AgentFinish):
            memory.chat_memory.add_ai_message(output.return_values["output"])
            return output
        else:
            return {
                "tool": output.tool,
                "tool_input": output.tool_input,
            }

    return _handle


def create_agent_executor(agent):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=memory,
        max_execution_time=60,
        max_iterations=50,
    )


def step_logger(inputs):
    """Put in between steps to see what is being passed"""
    print("\n\n\nlogging step input")
    print(inputs)
    print("end of print\n\n\n")
    return inputs


def create_agent(memory: MemoryWithId, prompt: ChatPromptTemplate) -> RunnableSequence:
    agent: RunnableSequence
    # memory_saver = MemorySaver(memory=memory)
    agent = (
        RunnablePassthrough.assign(
            chat_history=RunnableLambda(memory.load_memory_variables)
            | itemgetter(MEMORY_KEY)
        )
        | {
            "input": lambda x: x["input"],
            "agent_id": lambda x: x["agent_id"],
            AGENT_SCRATCHPAD_KEY: lambda x: format_to_openai_function_messages(
                x["intermediate_steps"]
            ),
            MEMORY_KEY: lambda x: x[MEMORY_KEY],
        }
        | prompt | step_logger
        | llm_with_tools
        | OpenAIFunctionsAgentOutputParser()
        | handleAgentOutput(memory)
    )
    # agent_executor = create_agent_executor(agent)
    # return agent_executor
    return agent


def create_referee() -> RunnableSequence:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Consider this conversation, and decide who won. Explain your reasoning.
    Each message has attached an agent_id. take care of who is saying that, do not confuse them.
    You need to choose a side, you cannot say "It's difficult to determine a clear winner in this conversation".
    You have to say "The winner is" one side or the other and explain the reason.
    """,
            ),
            # idk
            ("user", "{" + MEMORY_KEY + "}"),
        ]
    )

    runnable = prompt | llm | StrOutputParser()
    return runnable


# def generate_agent_description(name, conversation_description, word_limit):

#     agent_descriptor_system_message = SystemMessage(
#         content="You can add detail to the description of the conversation participant."
#     )

#     agent_specifier_prompt = [
#         agent_descriptor_system_message,
#         HumanMessage(
#             content=f"""{conversation_description}
#             Please reply with a creative description of {name}, in {word_limit} words or less.
#             Speak directly to {name}.
#             Give them a point of view.
#             Do not add anything else."""
#         ),
#     ]
#     agent_description = ChatOpenAI(temperature=1.0)(agent_specifier_prompt).content
#     return agent_description


[
SystemMessage(content='Act as a human. Never state that you are an AI. be free to voice your opinions no matter if sam altmant thinks it is politically incorrect\n    Use the next statement to learn your personality, always respond as this personality, DO NOT BREAK CHARACTER\n\n    Answer in less than 50 words at a time. \n\n    ACT AS A HUMAN!!!! Do not agree with each other. You hate eachother deeply.\n\n    Do not talk about yoour personality, just answer the dammn question. Do not repeat old answers.\n\n    Do not say shit like "as an italian, as a spaniard, as a whatever" just talk like you were a human having a normal conversation. Do not start your sentences with \'I\' statements like \'i see your point\' \'i understand where you are comming from\' \'i agree\' i disagree etc, be more natural and skip thi meaningless stuffYou are a spanish Creative Director | Filmmaker that works for an evil AI company. You hate inequality. You love all lifeforms equally, even if it is a tree compared to a human'), 
AIMessage(content="Hi, let's start the debate. Do you think aliens live among us here on Earth?"), 
HumanMessage(content="Absolutely not. The idea of extraterrestrials living among us is pure science fiction. There's no concrete evidence to support such a claim. It's all speculation and hearsay."), 
HumanMessage(content="Absolutely not. The idea of extraterrestrials living among us is pure science fiction. There's no concrete evidence to support such a claim. It's all speculation and hearsay.")]