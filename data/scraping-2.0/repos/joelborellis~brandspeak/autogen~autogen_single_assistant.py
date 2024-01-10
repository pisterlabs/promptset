import os
import openai
from openai import OpenAI
from dotenv import load_dotenv
import autogen
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent
from tools.searchtool import Search

load_dotenv()

openai_model: str = os.environ.get("OPENAI_MODEL")
openai.api_key = os.environ.get("OPENAI_API_KEY")
# create client for OpenAI
client = OpenAI(api_key=openai.api_key)
search: Search = Search()  # get instance of search to query corpus

config_list_gpt4 = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gpt-4-1106-preview", "gpt-4-32k-0613"],
    },
)

gpt4_config = {
    "cache_seed": 42,  # change the cache_seed for different trials
    "temperature": 0,
    "config_list": config_list_gpt4,
    "timeout": 120,
}

# Function to perform a search of the brandpeak corpus
def brandspeak_retriever(message):
    print("calling brandspeak retriever")
    search_result = search.search_hybrid(message)
    return search_result

if __name__ == '__main__':
        
        # Retrieve an existing assistant already setup as an OpenAI Assistant
        # this is OpenAI Assistant stuff
        brandspeak_retriever_assistant = client.beta.assistants.retrieve(
                        assistant_id="asst_GSZF7jLrOJ6LNWCWJzz4snW9",
                        ) 
        
        planner_assistant = client.beta.assistants.retrieve(
                        assistant_id="asst_HOeqxoPgbXdIzhepX8tvmekl",
                        ) 

        # define the config including the tools that the assistant has access to
        # this will be used by the GPTAssistant Agent that is Shadow Retriever
        brandspeak_retriever_config = {
            "assistant_id": brandspeak_retriever_assistant.id,
            "tools": [
                {
                    "type": "function",
                    "function": brandspeak_retriever,
                }
                    ]
        }
        planner_assistant_config = {
            "assistant_id": planner_assistant.id,
        }

        # this is autogen stuff defining the agent that is going to be in the group
        brandspeak_retriever_agent = GPTAssistantAgent(
            name="BrandspeakRetriever",
            instructions=None,
            llm_config=brandspeak_retriever_config,
        )

        # this is autogen stuff defining the agent that is going to be in the group
        planner_agent = GPTAssistantAgent(
            name="Planner",
            instructions="""You are a Planner.  Suggest a plan. Revise the plan based on feedback from Admin until Admin approval.
                The plan you create will involve an BrandspeakRetriever which can execute searches of their respective corpus of data.
                Explain the plan first. Be clear which step is performed by an BrandspeakRetriever.""",
            llm_config=planner_assistant_config,
        )

        brandspeak_retriever_agent.register_function(
            function_map={
                "brandspeak_retriever": brandspeak_retriever,
            }
        )
        user_proxy = autogen.UserProxyAgent(
            name="Admin",
            system_message="A human admin. Interact with the planner to discuss the plan. Plan execution needs to be approved by this admin.",
            code_execution_config=False,
        )

        groupchat = autogen.GroupChat(agents=[user_proxy, brandspeak_retriever_agent], messages=[], max_round=10)
        manager = autogen.GroupChatManager(groupchat=groupchat, name="brandspeak_manager")

        print("initiating chat")

        user_proxy.initiate_chat(
            manager,
            message="""
            Create a bulletpoint list of the similarities and differences between the visions of Oracle, Microsoft and AWS with regard to their future directions with AI.
            """,
            silent=False
        )