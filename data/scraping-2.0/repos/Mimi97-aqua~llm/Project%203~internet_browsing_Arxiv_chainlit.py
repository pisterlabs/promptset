from langchain.agents import initialize_agent, Tool, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
from openai import OpenAI
from dotenv import load_dotenv
import os
import chainlit as cl
import time

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
openai = OpenAI(api_key=api_key)


@cl.on_chat_start
def start():
    llm = ChatOpenAI(temperature=0.5, streaming=True)
    tools = load_tools(["arxiv"])

    agent_chain = initialize_agent(
        tools,
        llm,
        max_iterations=10,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )

    cl.user_session.set("agent", agent_chain)


@cl.on_message
async def main(message):
    agent = cl.user_session.get("agent")  # type: AgentExecutor
    cb = cl.LangchainCallbackHandler(stream_final_answer=True)

    # Get user input/prompt
    user_question = message.content

    action = 'arxiv'
    action_input = user_question
    observation = "I'm interested in getting more information about it."

    while True:
        try:
            # Ensure message follow expected format
            formatted_message = (f"Action: {action},"
                                 f"Action Input: {action_input},"
                                 f" Observation: {observation}")

            # Using formatted message API request
            await cl.make_async(agent.run)(formatted_message, callbacks=[cb])
            break  # Break out of the loop if the request is successful
        except cl.OpenAIError as e:
            if e.code == 'rate_limit_exceeded':
                # If rate limit exceeded, wait for 20 seconds and retry
                print("Rate limit exceeded. Waiting for 20 seconds...")
                time.sleep(20)
            else:
                # Print the error details
                print(f"OpenAI Error: {e}")
                print(f"Error message: {e.message}")
                print(f"Error code: {e.code}")
                print(f"Error type: {e.type}")
                print(f"Error param: {e.param}")
                break
