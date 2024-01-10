from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
# import openai

# openai.api_key = '<OPENAI_KEY_HERE>'

def create_agents():
    config_list = config_list_from_json(
        "OAI_CONFIG_LIST",
        filter_dict={
            "model": ["gpt-4",  "gpt-4-0314",  "gpt4",  "gpt-4-32k",  "gpt-4-32k-0314",  "gpt-4-32k-v0314"], 
            # "model": ["gpt-3.5-turbo", "gpt-3.5-turbo-0613",  "gpt3.5-turbo",  "gpt-3.5-turbo-16k",  "gpt-3.5-turbo-16k-0613",  "gpt-3.5-turbo-16k-v0613"], 
        }
    )
    assistant = AssistantAgent(name="Assistant", llm_config={"config_list": config_list})
    user_proxy = UserProxyAgent(name="user_proxy",code_execution_config={"work_dir":"coding"})
    return assistant, user_proxy

def main():
    print('Welcome to AutoGen PlayGround!')
    assistant, user_proxy = create_agents()
    # the assistant receives a message from the user, which contains the task description
    user_proxy.initiate_chat(
        assistant,
        # message="""Plot a chart of NVDA and TESLA stock price change YTD.""",
        message="""Plot a chart of NVDA and TESLA stock price change YTD.""",
    )
    

if __name__ == '__main__':   
    main()
    


