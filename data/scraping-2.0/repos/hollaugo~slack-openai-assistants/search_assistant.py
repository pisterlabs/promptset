import os 
import json
import uuid 
from typing import Type
from dotenv import load_dotenv

#Slack Imports 
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

#Langchain Imports
from langchain.agents.openai_assistant import OpenAIAssistantRunnable
from langchain.agents import AgentExecutor
from langchain_core.agents import AgentFinish
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

load_dotenv()

#Get Tavily API Key
tavily_api_key = os.getenv('TAVILY_API_KEY')


# Initialize the Slack Bolt App with the bot token
app = App(token=os.environ.get("SLACK_BOT_TOKEN"))
client = app.client

#Create Tavily Search Tool
search = TavilySearchAPIWrapper()
tavily_tool = TavilySearchResults(api_wrapper=search, return_direct=True, verbose=True)
search_input = {"query": "What is the Tesla stock price as of today?"}  
#output = tavily_tool.invoke(search_input)

# Print the output
#print("Output:", output)


#Function to create assistant
def create_assistant(name, instructions, tools, model, as_agent):
    assistant = OpenAIAssistantRunnable.create_assistant(
        name=name,
        instructions=instructions,
        tools=tools,
        model=model,
        as_agent=as_agent
    )
    assistant_id = assistant.assistant_id
    
    return assistant_id


assistant_id = "asst_Y7UMAfKqysNF0DnAqwnwlqbo"

#Function to use existing assistant
def existing_assistant(assistant_id):
    agent = OpenAIAssistantRunnable(assistant_id=assistant_id, as_agent=True)
    return agent

#Initialize Assistant and Tools
assistant = existing_assistant(assistant_id)
tools = [tavily_tool]

# Agent executor function for executing agents
def execute_agent(agent, tools, input):
    tool_map = {tool.name: tool for tool in tools}
    response = agent.invoke(input)
    while not isinstance(response, AgentFinish):
        tool_outputs = []
        for action in response:
            tool_output = tool_map[action.tool].invoke(action.tool_input)
            if isinstance(tool_output, list):  # Check if the output is a list
                tool_output = json.dumps(tool_output)  # Serialize the list to a JSON string
            print(f"Tool: {action.tool}, Output Type: {type(tool_output)}, Output: {tool_output}")
            tool_outputs.append({"output": tool_output, "tool_call_id": action.tool_call_id})
        response = agent.invoke(
            {
                "tool_outputs": tool_outputs,
                "run_id": action.run_id,
                "thread_id": action.thread_id,
            }
        )
    return response


# Example usage
#response = execute_agent(assistant, tools, {"content": "@<U05K6HV02UEI>: Send a message to the general channel saying hi"})
#print(response)
#print(response.return_values["output"])

# Listen and handle messages
@app.message("")
def message_handler(message, say, ack):
    ack()
    print(message)
    user_query = message['text']
    from_user = message['user']
    response = execute_agent(assistant, tools, {"content": user_query})
    ai_response = response.return_values["output"]
    print(ai_response)
    say(ai_response, thread_ts=message['ts'])


# Start your app
if __name__ == "__main__":
    SocketModeHandler(app, os.environ.get("SLACK_APP_TOKEN")).start()