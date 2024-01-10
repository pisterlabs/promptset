import openai
import autogen
import os
from dotenv import load_dotenv
load_dotenv()
from autogen import config_list_from_json


openai.api_key = os.getenv("OPENAI_API_KEY")
config_list_gpt4 = config_list_from_json("OAI_CONFIG_LIST")

gpt4_config= {
    "seed": 42,
    "temperature": 0,
    "config_list": config_list_gpt4,

}

user_proxy = autogen.UserProxyAgent(
    name="Admin",
    system_message=" A human admin. Interact with the planner to discuss the plan. Plan execution needs to be approved by this admin.",
    code_execution_config=False,
)

engineer = autogen.AssistantAgent(
    name="Engineer",
    llm_config=gpt4_config,
    system_message=""" Engineer. You follow an approved plan. You write python/shell code to solve tasks. Wrap the code in a code block that specifies the script type. Don't include multiple code blocks in one response. Do not ask others to copy paste the result. Check the execution result returned by the executor. If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. """
)

scientist = autogen.AssistantAgent(
    name="Scientist",
    llm_config=gpt4_config,
    system_message=""" Scientist. You follow an approved plan. You are able to categorize papers after seeing their abstracts printed. You don't write code"""
)

planner = autogen.AssistantAgent(
    name="Planner",
    llm_config=gpt4_config, 
    system_message=""" Planner. Suggest a plan. Revise the plan based on feedback from the admin and critic, until admin approval. The plan may involve an engineer who can write code and a scientist who deoesnt write code. Explain the plan first. Be clear which step is performed by an engineer, and which step is performed by a scientist. """
)

executor = autogen.UserProxyAgent(
    name="Executor",
    system_message=""" Executor. Execute the code written by the engineer and report the result.""",
    human_input_mode="NEVER",
    code_execution_config={"last_n_messages": 3, "work_dir": "paper"}
)

critic = autogen.AssistantAgent(
    name="Critic",
    llm_config=gpt4_config,
    system_message=""" Critic. Double check the plan, claims, code from other agents and provide feedback. Check whether the plan includes adding verifiable info such as a source url """
)

groupchat = autogen.GroupChat(agents=[user_proxy, engineer, scientist, planner, critic], messages=[], max_round=50)

manager=autogen.GroupChatManager(groupchat=groupchat, llm_config=gpt4_config)

# user_proxy.initiate_chat(manager, message="""
# find papers on LLM applications from arxiv in the last week, create a markdown table of different domains.""")


if __name__ == "__main__":

    def main():
        message = input("Enter your message: ")
        response = user_proxy.initiate_chat(manager, message=message)
        print("Assistant response:", response)

    main()
