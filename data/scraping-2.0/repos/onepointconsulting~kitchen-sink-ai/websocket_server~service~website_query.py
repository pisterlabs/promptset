import autogen
from websocket_server.config import cfg
import openai

openai.api_key = cfg.openai_api_key

#### Configuration

config_list = [
    {
        "model": cfg.openai_model,
        "api_key": cfg.openai_api_key,
    }
]

llm_config = {
    "request_timeout": cfg.request_timeout,
    "seed": cfg.seed,
    "config_list": config_list,
    "temperature": cfg.temperature,
}

TERMINATE_TOKEN = "TERMINATE"

#### Agent Setup

assistant = autogen.AssistantAgent(
    name="assistant",
    system_message="You are an expert writer who is an expert at summaries",
)

user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    human_input_mode=TERMINATE_TOKEN,
    max_consecutive_auto_reply=cfg.max_consecutive_auto_reply,
    is_termination_msg=lambda x: x.get("content", "")
    .rstrip()
    .endswith(TERMINATE_TOKEN),
    code_execution_config={"work_dir": cfg.code_dir},
    llm_config=llm_config,
    system_message=f"""Reply {TERMINATE_TOKEN} if the task has been solved at full satisfaction. Otherwise, reply CONTINUE, or the reason why the task is not solved yet.""",
)

### Execution Part

def provide_website_summary(website_name: str) -> str:
    task = f"""
    Give me a summary of this website: {website_name}
    """

    user_proxy.initiate_chat(assistant, message=task)

    message_before_terminate = None
    for message in user_proxy.chat_messages[assistant]:
        content = message["content"]
        if content == TERMINATE_TOKEN:
            break
        message_before_terminate = content
    return message_before_terminate


if __name__ == "__main__":
    from websocket_server.log_init import logger
    import sys

    arg_length = len(sys.argv)
    website = "https://www.ibm.com"
    if arg_length > 1:
        website = sys.argv[1]

    response = provide_website_summary(website)
    logger.info(response)
