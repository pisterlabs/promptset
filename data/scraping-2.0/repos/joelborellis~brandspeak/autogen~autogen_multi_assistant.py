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
search_o: Search = Search()  # get instance of search to query corpus

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

# Function to perform a Microsoft
def microsoft_retrieval(query):
    print("calling Microsoft search")
    search_result = search_m.search_hybrid(query, "Microsoft")
    return search_result

# Function to perform a AWS
def aws_retrieval(query):
    print("calling AWS search")
    search_result = search_a.search_hybrid(query, "AWS")
    return search_result

# Function to perform a Oracle
def oracle_retrieval(query):
    print("calling AWS search")
    search_result = search_o.search_hybrid(query, "Oracle")
    return search_result

if __name__ == '__main__':
        
        # Retrieve an existing assistant already setup as an OpenAI Assistant
        # this is OpenAI Assistant stuff
        microsoft_retriever = client.beta.assistants.retrieve(
                        assistant_id="asst_dANZJ31IHkMbvEHU0HDH3Zxt",
                        ) 
        
        aws_retriever = client.beta.assistants.retrieve(
                        assistant_id="asst_ekigD4AX2R3ij6tqTIYvtiVa",
                        ) 
        
        oracle_retriever = client.beta.assistants.retrieve(
                        assistant_id="asst_YBi5VOql08zBLzR3g7A4NnrN",
                        )
        
        linkedin_writer = client.beta.assistants.retrieve(
                        assistant_id="asst_VxerQBWNNWuuRMqqa0V716aw",
                        )
        
        planner_assistant = client.beta.assistants.retrieve(
                        assistant_id="asst_HOeqxoPgbXdIzhepX8tvmekl",
                        ) 

        # define the config including the tools that the assistant has access to
        # this will be used by the GPTAssistant Agent that is Shadow Retriever
        microsoft_retriever_config = {
            "assistant_id": microsoft_retriever.id,
            "tools": [
                {
                    "type": "function",
                    "function": microsoft_retrieval,
                }
                    ]
        }

        aws_retriever_config = {
            "assistant_id": aws_retriever.id,
            "tools": [
                {
                    "type": "function",
                    "function": aws_retrieval,
                }
                    ]
        }

        oracle_retriever_config = {
            "assistant_id": oracle_retriever.id,
            "tools": [
                {
                    "type": "function",
                    "function": oracle_retrieval,
                }
                    ]
        }

        linkedin_writer_config = {
            "assistant_id": linkedin_writer.id,
        }

        planner_assistant_config = {
            "assistant_id": planner_assistant.id,
        }


        # this is autogen stuff defining the agent that is going to be in the group
        microsoft_retriever_agent = GPTAssistantAgent(
            name="MicrosoftRetriever",
            instructions=None,
            llm_config=microsoft_retriever_config,
        )

        # this is autogen stuff defining the agent that is going to be in the group
        aws_retriever_agent = GPTAssistantAgent(
            name="AWSRetriever",
            instructions=None,
            llm_config=aws_retriever_config,
        )

        # this is autogen stuff defining the agent that is going to be in the group
        oracle_retriever_agent = GPTAssistantAgent(
            name="OracleRetriever",
            instructions=None,
            llm_config=oracle_retriever_config,
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
                The plan may involve a MicrosoftRetriever who can retrieve documents related to Microsoft and a AWSRetriever who can retrieve documents related to AWS, and an OracleRetriever who can retrieve documents related to Oracle.
                Only use these MicrosoftRetriever, AWSRetriever and OracleRetriever once to retrieve documents throughout to execution of the plan.
                The plan may involve a LinkedInWriter who is an expert at writing LinkedIn posts based on the information provided by the MicrosoftRetriever, AWSRetriever and OracleRetriever.
                Explain the plan first. Be clear which step is performed by a MicrosoftRetriever, and which step is performed by a AWSRetriever, and which step is performed by a OracleRetriever, and which step is performed by the LinkedInWriter.""",
            llm_config=planner_assistant_config,
        )

        microsoft_retriever_agent.register_function(
            function_map={
                "microsoft_retrieval": microsoft_retrieval,
            }
        )

        aws_retriever_agent.register_function(
            function_map={
                "aws_retrieval": aws_retrieval,
            }
        )

        oracle_retriever_agent.register_function(
            function_map={
                "oracle_retrieval": oracle_retrieval,
            }
        )

        user_proxy = autogen.UserProxyAgent(
            name="Admin",
            system_message="A human admin. Interact with the planner to discuss the plan. Plan execution needs to be approved by this Admin.",
            code_execution_config=False,
        )


        groupchat = autogen.GroupChat(agents=[user_proxy, microsoft_retriever_agent, aws_retriever_agent, oracle_retriever_agent,  planner_agent, linkedin_writer_agent], messages=[], max_round=20)
        manager = autogen.GroupChatManager(groupchat=groupchat, name="brandspeak_manager")

        print("initiating chat")

        user_proxy.initiate_chat(
            manager,
            message="""
            Write a LinkedIn article that discusses what AWS, Microsoft and Oracle are doing to enable enterprises to take advantage of AI solutions.  Use the information from recent keynote speeches at their annual conferences.
            """,
            silent=False
        )