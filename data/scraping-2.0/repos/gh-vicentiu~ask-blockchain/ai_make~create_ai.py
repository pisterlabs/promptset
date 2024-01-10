# filename ai-make/create_ai.py
import openai
import json
import logging

client = openai.Client()

def read_instructions(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def create_assistant(agent=None):
    from ai_tools.main_tools import tools_list
    from ai_tools.secondary_tools import tools_lite
    from ai_tools.route_tools import tools_route
    tool_list = tools_list
    tool_lite = tools_lite
    tool_route = tools_route

    if agent == "relay":
        instructions_relay = read_instructions("ai_make/instructions_relay.txt")
        assistant = client.beta.assistants.create(
            name=agent,
            instructions=instructions_relay,
            tools=tool_list,
            model="gpt-3.5-turbo-1106"        
        )
    elif agent == "agent_webhook":
        instructions_agent_webhook = read_instructions("ai_make/instructions_agent_webhook.txt")
        assistant = client.beta.assistants.create(
            name=agent,
            instructions=instructions_agent_webhook,
            tools=tool_route,
            model="gpt-3.5-turbo-1106"        
        )
    elif agent == "agent_coder":
        instructions_agent_coder = read_instructions("ai_make/instructions_agent_coder.txt")
        assistant = client.beta.assistants.create(
            name=agent,
            instructions=instructions_agent_coder,
            tools=tools_lite,
            model="gpt-3.5-turbo-1106"
        )
    else:
        logging.info(agent)
        raise ValueError("Invalid agent specified")

    return assistant

if __name__ == "__main__":
    assistant = create_assistant()
    print(f"Assistant created: {assistant}")
