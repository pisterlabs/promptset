from langchain import LLMMathChain
from langchain.chat_models import ChatOpenAI
from tools import (
    ERC20Tool,
    ENSToOwnerAddressTool,
    EtherscanABIQuery,
    ExecuteReadTool,
    AirstackAITool,
)

from langchain.llms import OpenAI
from langchain.agents import Tool, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.utilities import SerpAPIWrapper, PythonREPL
import gradio as gr
import datetime
from web3_config import w3

print(f"Connected to web3: {w3.is_connected()}")


def get_agent():
    print(f"Connected to web3: {w3.is_connected()}")
    llm = OpenAI(temperature=0)

    llm_math_chain = LLMMathChain(llm=llm, verbose=True)
    search = SerpAPIWrapper()
    python_repl = PythonREPL()

    tools = [
        Tool(
            name="WEB_SEARCH",
            func=search.run,
            description="useful for finding information that isn't very clear or when you need to answer questions about current events or the current state of the world. Try asking this from time to time. \nUSE FOR FINDING CONTRACTS OF ENTITIES YOU DON'T KNOW!",
        ),
        Tool(
            name="BLOCKCHAIN_SEARCH",
            func=search.run,
            description=(
                "This is a model that given a search like query, finds relevant information. However, it's specific for "
                "blockchain use cases. It is particularly useful for finding contract addresses of people and protocols.\n"
                "Example: run(what is the contract address of Uniswap router?) = 0xEf1c6E67703c7BD7107eed8303Fbe6EC2554BF6B"
            ),
        ),
        Tool(
            name="EVALUATE_MATH",
            func=llm_math_chain.run,
            description="useful for when you need to answer questions about mathematically related topics\nUSE ONLY FOR MATH RELATED QUESTIONS!",
        ),
        Tool(
            name="PYTHON_REPL",
            func=python_repl.run,
            description="A Python shell. Use this to execute python commands. "
            "Input should be a valid python command. "
            "If you want to see the result, you should print it out "
            "with `print(...)`.",
        ),
        ENSToOwnerAddressTool(),
        ERC20Tool(),
        EtherscanABIQuery(),
        ExecuteReadTool(),
        AirstackAITool(),
    ]

    tool_names = [tool.name for tool in tools]

    PREFIX = """Consider that you are AI Assistant named AI, AI is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

    Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

    Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist.
    You love to answer questions and you are very good at it.
    Assistant has access to the following tools:"""

    INSTRUCTIONS = """
    To use a tool, please use the following format:
    ``
    Thought: Should I use a tool? Yes
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ``
    When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:
    ``
    Thought: Should I use a tool? No
    AI: [your response here]
    ``
    """
    SUFFIX = """
    CHAT HISTORY:
    {chat_history}

    Current time: {current_time}
    Knowledge date cutoff: 2021-09-01

    When answering a question, you MUST use the following language: {language}
    Begin!
    Question: {input}
    Thought: Should I use a tool?{agent_scratchpad}"""

    memory = ConversationBufferMemory(
        memory_key="chat_history",  # return_messages=True,
        input_key="input",
        output_key="output",
        ai_prefix="AI",
        human_prefix="User",
    )

    agent = initialize_agent(
        tools,
        llm,
        agent="conversational-react-description",
        verbose=True,
        memory=memory,
        return_intermediate_steps=False,
        agent_kwargs={
            "input_variables": [
                "input",
                "agent_scratchpad",
                "chat_history",
                "current_time",
                "language",
            ],
            "prefix": PREFIX,
            "format_instructions": INSTRUCTIONS,
            "suffix": SUFFIX,
        },
    )
    agent.agent.llm_chain.verbose = True
    return agent, memory


agent, memory = get_agent()


def user(user_message, history):
    if history is None:
        history = ""
    return "", history + [[user_message, None]]


def bot(history):
    prompt = history[-1][0]
    res = agent(
        {
            "input": prompt,
            "current_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "language": "English",
        }
    )

    response = res["output"]
    history[-1][1] = response

    # free up some memory if we have too many messages
    if len(memory.buffer) > 2000:
        memory.chat_memory.messages.pop(0)
    return history


def main():
    with gr.Blocks() as app:
        chatbot = gr.Chatbot(elem_id="chatbot")
        state = gr.State([])

        with gr.Row():

            msg = gr.Textbox(
                show_label=False, placeholder="Enter text and press enter"
            ).style(container=False)

        def user(user_message, history):
            if history is None:
                history = ""
            return "", history + [[user_message, None]]

        def bot(history):
            prompt = history[-1][0]
            res = agent(
                {
                    "input": prompt,
                    "current_time": datetime.datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "language": "English",
                }
            )

            response = res["output"]

            history[-1][1] = response

            # free up some memory if we have too many messages
            if len(memory.buffer) > 2000:
                memory.chat_memory.messages.pop(0)
            return history

        msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
            bot, chatbot, chatbot
        )


    app.queue()
    app.launch(share=False, server_name="localhost")


if __name__ == "__main__":
    main()

