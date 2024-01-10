import os
import openai
from openai import OpenAI
from dotenv import load_dotenv
import autogen
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent

load_dotenv()

openai_model: str = os.environ.get("OPENAI_MODEL")
openai.api_key = os.environ.get("OPENAI_API_KEY")
# create client for OpenAI
client = OpenAI(api_key=openai.api_key)


if __name__ == '__main__':
        file_stats = client.files.create(
            file=open("./data/nfl_offensive_stats.csv", "rb"),
            purpose='assistants'
        )

        file_map = client.files.create(
            file=open("./data/nfl_offensive_stats_mapping.csv", "rb"),
            purpose='assistants'
        )

        file_map_teams = client.files.create(
            file=open("./data/nfl_offensive_stats_mapping_teams.csv", "rb"),
            purpose='assistants'
        )

        coder_assistant = client.beta.assistants.create(
            name="Python Developer",
            instructions="You are a python developer",
            model="gpt-4-1106-preview",
            tools = [ { "type": "code_interpreter" } ],
            file_ids=[file_stats.id, file_map.id, file_map_teams.id]
        )

        analyst_assistant = client.beta.assistants.create(
            name="Data Analyst",
            instructions="You are a data analyst",
            model="gpt-4-1106-preview",
            tools = [ { "type": "code_interpreter" } ],
            file_ids=[file_stats.id, file_map.id, file_map_teams.id]
        )

        coder_llm_config = {
            "assistant_id": coder_assistant.id
        }

        analyst_llm_config = {
            "assistant_id": analyst_assistant.id
        }

        coder = GPTAssistantAgent(
            name="Coder_Assistant",
            instructions="""
            You are an expert at writing python code to solve problems. 
            Reply TERMINATE when the task is solved and there is no problem
            """,
            llm_config=coder_llm_config
        )

        analyst = GPTAssistantAgent(
            name="Data_Analyst",
            instructions="""
            You are a data analyst that offers insight into data.
            """,
            llm_config=analyst_llm_config,
        )

        user_proxy = autogen.UserProxyAgent(
            name="UserProxy",
            code_execution_config={
                "work_dir" : "coding",
            },
            system_message="Admin"
        )

        groupchat = autogen.GroupChat(agents=[user_proxy, coder, analyst], messages=[], max_round=10)
        manager = autogen.GroupChatManager(groupchat=groupchat)

        user_proxy.initiate_chat(
            manager,
            message="""
            What are the player trends.
            Give me an overview of the data. 
            Show the code you used to solve it.
            """
        )