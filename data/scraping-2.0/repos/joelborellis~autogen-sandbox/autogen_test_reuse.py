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
        
        # Retrieve an existing assistant
        coder_assistant = client.beta.assistants.retrieve(
                        assistant_id="asst_BD6LjCitqFEzRIatvu2FNW7X",
                        ) 

        # Retrieve an existing assistant
        analyst_assistant = client.beta.assistants.retrieve(
                        assistant_id="asst_WLmw3TELCkX7tXUyjVxFskzB",
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
            Does Najee Harris perfoem well against the Cleveland Browns.
            """
        )