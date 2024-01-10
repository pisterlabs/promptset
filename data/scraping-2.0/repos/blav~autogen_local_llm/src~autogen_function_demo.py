from model_config import get_config
from llama_cpp import Llama
from openai_api import start_openai_api_thread
from autogen.agentchat import AssistantAgent, UserProxyAgent
if __name__ == "__main__":
    config = get_config("Mistral-7B-Instruct-v0.1-function-calling-v2")
    
    start_openai_api_thread(Llama(**config))

    llm_config={
        "cache_seed": None,
        "config_list": [{
            "base_url": "http://localhost:8000/v1",
            "model": "gpt-4",
            "api_key": "dontcare",
        }],
        "tools": [{
            "type": "function",
            "function": {
                "name": "weather",
                "description": "Get weather information for a location.",
                "parameters": {
                    "type": "object",
                    "title": "weather",
                    "properties": {
                        "location": {
                            "title": "location",
                            "type": "string"
                        },
                    },
                    "required": [ 
                        "location", 
                    ]
                }
            }
        }, {
            "type": "function",
            "function": {
                "name": "traffic",
                "description": "Get traffic information for a location and date.",
                "parameters": {
                    "type": "object",
                    "title": "traffic",
                    "properties": {
                        "location": {
                            "title": "location",
                            "type": "string"
                        },
                        "date": {
                            "title": "date",
                            "type": "string"
                        },
                    },
                    "required": [ 
                        "location", 
                        "date",
                    ]
                }
            }
        }],
    }

    chatbot = AssistantAgent(
        name="chatbot",
        system_message="General purpose life helping chatbot.",
        llm_config=llm_config,
    )

    user_proxy = UserProxyAgent(
        name="human",
        is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="ALWAYS",
        max_consecutive_auto_reply=8,
        code_execution_config={"work_dir": "coding"},
    )

    user_proxy.register_function(
        function_map={
            "weather": lambda location: f"weather is nice".format(location=location),
            "traffic": lambda location, date: f"busy".format(location=location, date=date)
        }
    )

    user_proxy.initiate_chat(
        chatbot,
        message="how's the weather in Tokyo today?",
    )