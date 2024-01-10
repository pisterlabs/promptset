import os
import openai
from openai import OpenAI
from dotenv import load_dotenv
import autogen
from autogen.agentchat.contrib.gpt_assistant_agent import GPTAssistantAgent
from backend.tools.searchtool import Search

load_dotenv()

openai_model: str = os.environ.get("OPENAI_MODEL")
# create client for OpenAI
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

config_list_gpt4 = autogen.config_list_from_json(
    "OAI_CONFIG_LIST",
    filter_dict={
        "model": ["gpt-4-1106-preview", "gpt-4"],
    },
)

gpt4_config = {
    "cache_seed": 42,  # change the cache_seed for different trials
    "temperature": 0,
    "config_list": config_list_gpt4,
    "timeout": 120,
}

# Function to perform a Shadow Search
def generic_retriever(query, index):
    print(f"calling search with - {query} - {index}")
    search: Search = Search(index)  # get instance of search to query corpus using the name of the index
    search_result = search.search_hybrid(query)
    return search_result

if __name__ == '__main__':
        
        # Retrieve an existing assistant already setup as an OpenAI Assistant
        # this is OpenAI Assistant stuff
        retriever_assistant = client.beta.assistants.retrieve(
                        assistant_id="asst_CqfJXxZLQk6xv2zNzVGU0zVj",
                        ) 
        
        # Retrieve an existing assistant already setup as an OpenAI Assistant
        # this is OpenAI Assistant stuff
        #planner_assistant = client.beta.assistants.retrieve(
        #                assistant_id="asst_7LH25ZRiZXMk05J7F9NkyDSY",
        #                ) 

        # define the config including the tools that the assistant has access to
        # this will be used by the GPTAssistant Agent that is Shadow Retriever
        retriever_config = {
            "assistant_id": retriever_assistant.id,
            "tools": [
                {
                    "type": "function",
                    "function": generic_retriever,
                }
                    ]
        }

        # define the config including the tools that the assistant has access to
        # this will be used by the GPTAssistant Agent that is Shadow Retriever
        #planner_config = {
        #    "assistant_id": planner_assistant.id,
        #}

        # this is autogen stuff defining the agent that is going to be in the group
        generic_retriever_agent = GPTAssistantAgent(
            name="GenericRetriever",
            llm_config=retriever_config,
        )

        generic_retriever_agent.register_function(
            function_map={
                "generic_retriever": generic_retriever,
            }
        )

        # this is autogen stuff defining the agent that is going to be in the group
        #planner = GPTAssistantAgent(
        #    name="GenericPlanner",
        #    llm_config=planner_config,
        #    instructions='''Planner. Suggest a plan. Revise the plan based on feedback from admin, until admin approval.
        #        The plan may involve GenericRetriever who can retrieve data.
        #        Explain the plan first. Be clear which step is performed by GenericRetriever.
        #        '''
        #)

        # this is autogen stuff defining the agent that is going to be in the group
        planner = autogen.AssistantAgent(
            name="Planner",
            system_message='''Planner. Suggest a plan. Revise the plan based on feedback from admin, until admin approval.
                The plan may involve retriever who can retrieve data.
                Explain the plan first. Be clear which step is performed by a retriever.
                ''',
            llm_config=gpt4_config,
        )

        user_proxy = autogen.UserProxyAgent(
            name="Admin",
            system_message="A human admin. Interact with the planner to discuss the plan. Plan execution needs to be approved by this admin."
        )

        groupchat = autogen.GroupChat(agents=[user_proxy, generic_retriever_agent, planner], messages=[], max_round=10)
        manager = autogen.GroupChatManager(groupchat=groupchat)

        print("initiating chat")

        user_proxy.initiate_chat(
            manager,
            message="""
            I have a first meeting with a prospect United Healthcare - what do I need to find out and what are the most important things I need to relate to them.  Use the index called sales_vector_index.
            """
        )