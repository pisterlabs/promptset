from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from constant.prompt import critic_prompt
from util.ansi_console_utils import (
    print_critic_agent_human_message,
    print_critic_agent_ai_message,
    print_error_message,
)
from util.json_utils import fix_and_parse_json


class CriticAgent:
    def __init__(
        self,
        model_name: str,
        request_timout: int,
    ):
        self.llm = ChatOpenAI(
            model_name=model_name,
            temperature=0,
            request_timeout=request_timout,
        )

    def render_system_message(self) -> SystemMessage:
        system_message = SystemMessage(content=critic_prompt.content)
        return system_message

    def render_human_message(
        self, error_message: str, output: str, task: str
    ) -> HumanMessage:
        observation = ""
        if error_message:
            observation += f"Execution error: {error_message}\n\n"
        else:
            observation += f"Execution error: No error\n\n"
        if output:
            observation += f"Execution log: {output}\n\n"
        observation += f"Task: {task}\n\n"
        print_critic_agent_human_message(message_content=observation)
        return HumanMessage(content=observation)

    def ai_check_task_success(
        self, messages: list[BaseMessage], max_retries: int
    ) -> tuple[str, bool]:
        if max_retries == 0:
            print_error_message(message_content="最大重试次数已用完, 无法解析判别代理的AI输出")
            return False, ""
        critic = self.llm(messages).content
        print_critic_agent_ai_message(message_content=critic)
        try:
            response = fix_and_parse_json(critic)
            assert response["success"] in [True, False]
            if "critique" not in response:
                response["critique"] = ""
            return response["critique"], response["success"]
        except Exception as e:
            error_message = f"解析判别代理的AI输出时出错: {e}, 正在重试..."
            print_error_message(message_content=error_message)
            return self.ai_check_task_success(
                messages=messages,
                max_retries=max_retries - 1,
            )

    def check_task_success(
        self, error_message: str, output: str, task: str, max_retries: int
    ) -> tuple[str, bool]:
        messages = [
            self.render_system_message(),
            self.render_human_message(
                error_message=error_message, output=output, task=task
            ),
        ]
        return self.ai_check_task_success(messages=messages, max_retries=max_retries)
