from autogen import oai
from autogen import AssistantAgent, UserProxyAgent
import openai


oai.ChatCompletion.start_logging()

# Configure OpenAI settings
openai.api_type = "openai"
openai.api_key = "..."
openai.api_base = "http://127.0.0.1:5552/v1"
openai.api_version = "2023-11-22"


# create a text completion request
response = oai.Completion.create(
    config_list=[
        {
            "model": "Yi34B200K-Llamafied-chatSFT-fcv3-GPTQ",
            "base_url": "http://127.0.0.1:5552/v1",
            "api_type": "openai",
            "api_key": "...", # just a placeholder
        }
    ],
    prompt="Hi",
)
print(response)

# # create a chat completion request
# response = oai.ChatCompletion.create(
#     config_list=[
#         {
#             "model": "Yi-34B-200K-Llamafied-chat-SFT-function-calling-v3-GPTQ",
#             "base_url": "http://127.0.0.1:5552/v1",
#             "api_type": "openai",
#             "api_key": "...",
#         }
#     ],
#     messages=[{"role": "user", "content": "Hi"}]
# )
# print(response)


local_config_list = [
    {
        'model': 'Yi-34B-200K-Llamafied-chat-SFT-function-calling-v3-GPTQ',
        'api_key': '...',
        'api_type': 'openai',
        'api_base': "http://127.0.0.1:5552/v1",
        'api_version': '2023-11-22'
    }
]

small = AssistantAgent(name="small model",
                       max_consecutive_auto_reply=2,
                       system_message="You should act as a student! Give response in 2 lines only.",
                       llm_config={
                           "config_list": local_config_list,
                           "temperature": 0.5,
                       })

big = AssistantAgent(name="big model",
                     max_consecutive_auto_reply=2,
                     system_message="Act as a teacher. Give response in 2 lines only.",
                     llm_config={
                         "config_list": local_config_list,
                         "temperature": 0.5,
                     })

big.initiate_chat(small, message="Who are you?")

