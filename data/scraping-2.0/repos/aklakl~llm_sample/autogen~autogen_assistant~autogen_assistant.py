import autogen
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
import openai


#=============autogen configuration=============
# Load LLM inference endpoints from an env variable or a file
# See https://microsoft.github.io/autogen/docs/FAQ#set-your-api-endpoints
# and OAI_CONFIG_LIST_sample
# config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")
# refer:https://github.com/microsoft/autogen

#Custom configurations
#base_url = "http://localhost:1234/v1"  # You can specify API base URLs if needed. eg: localhost:8000
#http://10.9.150.174:1234/v1/models
base_url = "http://192.168.40.229:1234/v1"  #Sidan's Server Mistral-7B-Instruct-v0.1-GGUF/mistral-7b-instruct-v0.1.Q8_0.gguf
#base_url = "http://airedale-native-chicken.ngrok-free.app/v1"  # Seems like OpenLLM server only supput SKD is openai==0.28  pyautogen==0.1.14 . refer: https://colab.research.google.com/drive/1GKlfU7Fjq30oQPirHvCcQy_e_B8vNEDs?usp=sharing
api_type = "openai"  # Type of API, e.g., "openai" or "aoai".
api_version = None  # Specify API version if needed.
#api_model= "palm/chat-biso"
#api_model="NousResearch--Nous-Hermes-llama-2-7b"
api_model="TinyLlama--TinyLlama-1.1B-Chat-v1.0"
api_key = "sk-llllllllllllllllllllll"   #openai==0.28 key style must be like this,Even your local LLM doesn't use it also follow the key style. refer:https://pypi.org/project/openai/0.28.0/

config_list_structure = [
    {
        "model": "gpt-4",
        "api_key": "<your OpenAI API key here>"
    },
    {
        "model": "gpt-4",
        "api_key": "<your OpenAI API key here>"
    },
    {
        "model": "Llama-2-7B-Chat-GGML",
        "api_key": "<your Azure OpenAI API key here>",
        "base_url": "http://localhost:1234/v1",
        "api_type": "azure",
        "api_version": "2023-07-01-preview"
    },
    {
        "model": "<your Azure OpenAI deployment name>",
        "api_key": "<your Azure OpenAI API key here>",
        "base_url": "<your Azure OpenAI API base here>",
        "api_type": "azure",
        "api_version": "2023-07-01-preview"
    },
    {
        "model": "<your Azure OpenAI deployment name>",
        "api_key": "<your Azure OpenAI API key here>",
        "base_url": "<your Azure OpenAI API base here>",
        "api_type": "azure",
        "api_version": "2023-07-01-preview"
    }
]
log_msg = "Configuration List config_list_structure=>"
#print(f"{log_msg} {config_list_structure}")




# # refer: https://microsoft.github.io/autogen/docs/reference/oai/openai_utils/
# api_keys = []
# base_urls = []
# api_types = []
# api_versions =[]
# config_list2 = autogen.get_config_list(
#     api_keys=api_keys,
#     base_urls=base_urls,
#     api_type=api_types,
#     api_version=api_versions
# )


#Static config_list2 
config_list2 = [
    {
    "model":api_model,
    "base_url":base_url,
    "api_base":base_url,
    "api_version":api_version,
    "api_key":api_key
    }
]
log_msg = "Configuration List2:"
#print(f"{log_msg} {config_list2}")
#=============autogen configuration=============



#=============example with openai==0.28 completion =============
#refer:https://pypi.org/project/openai/0.28.0/
'''
This code required the dependency follow these(also pay attention with  api_key and api_model) :
pip install openai==0.28 
'''
def run_openai_completion():
    try:
        openai.api_key = api_key  # supply your API key however you choose
        openai.api_base= base_url # supply your api base URL If you have your own LLM
        completion = openai.ChatCompletion.create(model=api_model, messages=[{"role": "user", "content": "Who are you?"}])
        print(completion.choices[0].message.content)
    except Exception as e:
        print(
            f"""
            run_openai_completion failed with Exception{e}. \n"""
        )  
#=============example with openai completion =============





#=============autogen run_autogen_with_twoagent_pyautogen_latest_version=============
#refer:https://pypi.org/project/pyautogen/0.1.14/
'''
This code required the dependency follow these(also pay attention with api_base and api_key and api_model) :
pip install openai==0.28  pyautogen==0.1.14 
'''
def run_autogen_with_twoagent_pyautogen_latest_version():
    try:
        #Base on openai==0.28  pyautogen==0.1.14 ,If you want to use your own LLM,you must be override your openai.api_base and openai.api_key for autogen, otherwise won't be work.
        openai.api_key = api_key  # supply your API key however you choose
        openai.api_base= base_url # supply your api base URL If you have your own LLM
        assistant = AssistantAgent("assistant", llm_config={"api_key":api_key,"base_url": base_url,"api_model":api_model})
        user_proxy = UserProxyAgent("user_proxy", code_execution_config={"work_dir": "coding"})
        #user_proxy.initiate_chat(assistant, message="Show me the YTD gain of 10 largest technology companies as of today.")
        user_proxy.initiate_chat(assistant, message="Plot a chart of NVDA and TESLA stock price change YTD.")
        # This triggers automated chat to solve the task
    except Exception as e:
        print(f"""run_autogen_with_twoagent_pyautogen_latest_version failed with Exception{e}. \n""")   
# ##====================================================



#
##refer:https://medium.com/@krtarunsingh/mastering-autogen-a-comprehensive-guide-to-next-generation-language-model-applications-b375d9b4dc6d
# autogen.ChatCompletion.start_logging()
# response = autogen.Completion.create(
#  context={"problem": "How many positive integers, not exceeding 100, are multiples of 2 or 3 but not 4?"},
#  prompt="{problem} Solve the problem carefully.",
#  allow_format_str_template=True,
#  **config
# )
#autogen.ChatCompletion.print_usage_summary()
# autogen.ChatCompletion.stop_logging()




#=========================autogen with twoagent example success===========================
#refer:https://github.com/microsoft/autogen/blob/main/test/twoagent.py
# Load LLM inference endpoints from an env variable or a file
# See https://microsoft.github.io/autogen/docs/FAQ#set-your-api-endpoints
# and OAI_CONFIG_LIST_sample
'''
This code required the dependency follow these(also pay attention with config_list.api_base and config_list.api_key and config_list.api_model) :
pip install openai==0.28  pyautogen==0.1.14 
'''
def run_autogen_with_twoagent():
    config_list = [
        {
        "model":api_model,
        "base_url":base_url,
        "api_base":base_url,
        "api_version":api_version,
        "api_key":api_key
        }
    ]
    try:
        #config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST")
        assistant = AssistantAgent("assistant", llm_config={"config_list": config_list})
        user_proxy = UserProxyAgent("user_proxy", human_input_mode="NEVER", code_execution_config={"work_dir": "coding"})
        user_proxy.initiate_chat(assistant, message="Plot a chart of NVDA and TESLA stock price change YTD.")
    except Exception as e:
        print(
            f"""
            run_autogen_with_Assistant_andrun_autogen_with_twoagent_userProxy failed with Exception{e}. \n"""
        )  
#=========================autogen with twoagent example success===========================




#========================run_autogen_with_Assistant_and_userProxy============================
'''
This code required the dependency follow these(also pay attention with config_list.api_base and config_list.api_key and config_list.api_model) :
pip install openai==0.28  pyautogen==0.1.14 
'''
def run_autogen_with_Assistant_and_userProxy():
    try:
        # create an AssistantAgent named "assistant"
        assistant = autogen.AssistantAgent(
            name="assistant",
            llm_config={
                "cache_seed": 42,  # seed for caching and reproducibility
                "config_list": config_list2,  # a list of OpenAI API configurations
                "temperature": 0,  # temperature for sampling
            },  # configuration for autogen's enhanced inference API which is compatible with OpenAI API
        )
        # create a UserProxyAgent instance named "user_proxy"
        user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE"),
            code_execution_config={
                "work_dir": "coding",
                "use_docker": False,  # set to True or image name like "python:3" to use docker
            },
        )
        # the assistant receives a message from the user_proxy, which contains the task description
        user_proxy.initiate_chat(
            assistant,
            message="""What date is today? Compare the year-to-date gain for META and TESLA.""",
        )
    except Exception as e:
        print(
            f"""
            run_autogen_with_Assistant_and_userProxy failed with Exception{e}. \n"""
        )  # noqa
#========================run_autogen_with_Assistant_and_userProxy============================


#running
if __name__ == "__main__":
    #run_autogen_with_Assistant_and_userProxy()
    #run_autogen_with_twoagent()
    #run_openai_completion()
    run_autogen_with_twoagent_pyautogen_latest_version()