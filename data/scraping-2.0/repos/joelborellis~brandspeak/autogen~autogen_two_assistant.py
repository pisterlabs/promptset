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
search_m: Search = Search()  # get instance of search to query corpus
search_a: Search = Search()  # get instance of search to query corpus

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

# Function to perform a Shadow Search
def microsoft_retrieval(query):
    print("calling Microsoft search")
    search_result = search_m.search_hybrid(query, "Microsoft")
    return search_result

# Function to perform a Shadow Search
def aws_retrieval(query):
    print("calling AWS search")
    search_result = search_a.search_hybrid(query, "AWS")
    return search_result

if __name__ == '__main__':
        
        # Retrieve an existing assistant already setup as an OpenAI Assistant
        # this is OpenAI Assistant stuff
        linkedin_writer = client.beta.assistants.retrieve(
                        assistant_id="asst_VxerQBWNNWuuRMqqa0V716aw",
                        )
        
        planner_assistant = client.beta.assistants.retrieve(
                        assistant_id="asst_HOeqxoPgbXdIzhepX8tvmekl",
                        ) 

        # define the config including the tools that the assistant has access to
        # this will be used by the GPTAssistant Agent that is Shadow Retriever

        linkedin_writer_config = {
            "assistant_id": linkedin_writer.id,
        }

        planner_assistant_config = {
            "assistant_id": planner_assistant.id,
        }

        user_proxy = autogen.UserProxyAgent(
            name="Admin",
            system_message="A human admin. Interact with the planner to discuss the plan. Plan execution needs to be approved by this Admin.",
            code_execution_config=False,
        )


        # this is autogen stuff defining the agent that is going to be in the group
        linkedin_writer_agent = GPTAssistantAgent(
            name="LinkedInWriter",
            instructions=None,
            llm_config=linkedin_writer_config,
        )

        # this is autogen stuff defining the agent that is going to be in the group
        planner_agent = GPTAssistantAgent(
            name="Planner",
            instructions="""You are a Planner.  Suggest a plan. Revise the plan based on feedback from Admin until Admin approval.
                The plan may involve a LinkedInWriter who is an expert at writing LinkedIn posts.  LinkedInWriter has access to tools to search and retrieve documents needed for the task.
                Explain the plan first. Be clear which step is performed which steps are performed by the LinkedInWriter.""",
            llm_config=planner_assistant_config,
        )

        linkedin_writer_agent.register_function(
            function_map={
                "microsoft_retrieval": microsoft_retrieval,
                "aws_retrieval": aws_retrieval,
            }
        )

        groupchat = autogen.GroupChat(agents=[user_proxy, planner_agent, linkedin_writer_agent], messages=[], max_round=20)
        manager = autogen.GroupChatManager(groupchat=groupchat, name="brandspeak_manager")

        print("initiating chat")

        user_proxy.initiate_chat(
            manager,
            message="""
            Write a LinkedIn post that highlights key announcements from Microsoft Ignite and AWS re:Invent.  Be sure to capture any key product announcements and partnerships announced by the keynote speakers.
            """,
            silent=False
        )