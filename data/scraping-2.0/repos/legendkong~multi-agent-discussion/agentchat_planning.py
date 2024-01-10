import autogen
import openai
import json

# load OpenAI API key from config file
with open("OAI_CONFIG_LIST.json", "r") as f:
    config = json.load(f)
openai.api_key = config["api_key"]

# this function loads a list of configurations from an environment variable or a json file
config_list = autogen.config_list_from_json(
    "OAI_CONFIG_LIST", 
    filter_dict={
        "model": ["gpt-4", "gpt-4-0314", "gpt4", "gpt-4-32k", "gpt-4-32k-0314", "gpt-4-32k-v0314"],
    },
)

# Assistant AGENT comes with a default system message. You can overwrite it by specifying the "system_message" parameter as below.
planner = autogen.AssistantAgent(
    name="planner",
    llm_config={"config_list": config_list},
    # the default system message of the AssistantAgent is overwritten here
    system_message="You are a helpful AI assistant. You suggest coding and reasoning steps for another AI assistant to accomplish a task. Do not suggest concrete code. For any action beyond writing code or reasoning, convert it to a step which can be implemented by writing code. For example, the action of browsing the web can be implemented by writing code which reads and prints the content of a web page. Finally, inspect the execution result. If the plan is not good, suggest a better plan. If the execution is wrong, analyze the error and suggest a fix."
)
# This is the user role (proxy) for the planner agent. We are doing this because chat completions use "assistant" and "user" roles. So, we need to specify the user role for the planner agent as below. But this will not take in any user input. It will only send messages to the planner agent.
planner_user = autogen.UserProxyAgent(
    name="planner_user",
    max_consecutive_auto_reply=0,  # terminate without auto-reply
    human_input_mode="NEVER",
)

# serves as a bridge for communicating with planner
def ask_planner(message):
    planner_user.initiate_chat(planner, message=message)
    # return the last message received from the planner
    return planner_user.last_message()["content"]

# create an AssistantAgent instance named "assistant"
assistant = autogen.AssistantAgent(
    name="assistant",
    llm_config={
        "temperature": 0,
        "request_timeout": 600,
        "seed": 42,
        "model": "gpt-4",
        "config_list": autogen.config_list_openai_aoai(exclude="aoai"),
        "functions": [
            {
                "name": "ask_planner",
                "description": "ask planner to: 1. get a plan for finishing a task, 2. verify the execution result of the plan and potentially suggest new plan.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "question to ask planner. Make sure the question include enough context, such as the code and the execution result. The planner does not know the conversation between you and the user, unless you share the conversation with the planner.",
                        },
                    },
                    "required": ["message"],
                },
            },
        ],
    }
)

# create a UserProxyAgent instance named "user_proxy"
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode="TERMINATE",
    max_consecutive_auto_reply=5,
    # is_termination_msg=lambda x: "content" in x and x["content"] is not None and x["content"].rstrip().endswith("TERMINATE"),

    code_execution_config={"work_dir": "planning", "use_docker": True}, # Docker is set to true by default
    function_map={"ask_planner": ask_planner},
)

# the assistant receives a message from the user, which contains the task description
user_proxy.initiate_chat(
    assistant,
    message="""Suggest an improvement to the most popular repo in legendkong github. use subprocess for pip installs""", # add "use subprocess for pip install" to the end of the message if you are getting powershell error
)