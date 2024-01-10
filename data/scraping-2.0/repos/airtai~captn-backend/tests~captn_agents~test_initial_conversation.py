# from autogen import OpenAIWrapper
from typing import Any, Dict, Union
from unittest.mock import Mock

from captn.captn_agents.backend.initial_team import InitialTeam

from .utils import last_message_is_termination

response_prefix = "Response from team 'planning_team_1':\n"
create_team = Mock()
create_team.side_effect = [
    f"{response_prefix}should I distinguish between lower and upper case letters?",
    f"{response_prefix}The task has been done.",
] * 5


answer_to_team_lead_question = Mock()
answer_to_team_lead_question.side_effect = [
    f"{response_prefix}The task has been done.",
] * 5

function_map = {
    "create_team": create_team,
    "answer_to_team_lead_question": answer_to_team_lead_question,
}

roles = [
    {
        "Name": "User_proxy",
        "Description": "Your job is to comunicate with the Product Owner, do NOT suggest any code or execute the code by yourself",
    },
    {
        "Name": "Product_owner",
        "Description": "You are a product owner in a software company.",
    },
]
task = "Create a python program for checking whether a string is palindrome or not"

initial_team = InitialTeam(
    user_id=1,
    task=task,
    roles=roles,
    conv_id=13,
    # function_map=function_map,  # type: ignore
    human_input_mode="NEVER",
)


def test_inital_message() -> None:
    for key in ["## Guidelines", "## Constraints"]:
        assert key in initial_team.initial_message, key

    expected_commands = """## Commands
You have access to the following commands:
1. create_team: Create an ad-hoc team to solve the problem, params: (json_as_a_string: string)
2. answer_to_team_lead_question: Answer to the team leaders question, params: (answer: string, team_name: str)
"""
    assert expected_commands in initial_team.initial_message


# @pytest.mark.vcr(
#     filter_headers=["api-key"]
# )
def test_initial_team() -> None:
    initial_team.initiate_chat()

    assert last_message_is_termination(initial_team)

    # for name in logging.root.manager.loggerDict:
    #     print(name)
    # loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]


try:
    from termcolor import colored
except ImportError:

    def colored(x: str, *args: Any, **kwargs: Any) -> Any:  # type: ignore
        return x


def _print_received_message(
    message: Union[Dict[str, Any], str]
) -> None:  # , sender: Agent):
    # print the message received
    # print(colored(sender.name, "yellow"), "(to", f"{self.name}):\n", flush=True)
    # print(f"{message['name']}:", flush=True)
    if message.get("role") == "function":  # type: ignore
        func_print = f"***** Response from calling function \"{message['name']}\" *****"  # type: ignore
        print(colored(func_print, "green"), flush=True)
        print(message["content"], flush=True)  # type: ignore
        print(colored("*" * len(func_print), "green"), flush=True)
    else:
        content = message.get("content")  # type: ignore
        if content is not None:
            # if "context" in message:
            #     content = OpenAIWrapper.instantiate(
            #         content,
            #         message["context"],
            #         self.llm_config and self.llm_config.get("allow_format_str_template", False),
            #     )
            print(content, flush=True)
        if "function_call" in message:
            function_call = dict(message["function_call"])  # type: ignore
            func_print = f"***** Suggested function Call: {function_call.get('name', '(No function name found)')} *****"
            print(colored(func_print, "green"), flush=True)
            print(
                "Arguments: \n",
                function_call.get("arguments", "(No arguments found)"),
                flush=True,
                sep="",
            )
            print(colored("*" * len(func_print), "green"), flush=True)
    print("\n", "-" * 80, flush=True, sep="")


def test_get_messages() -> None:
    initial_team.initiate_chat()
    print("*" * 200)
    print("*" * 200)
    print("*" * 200)

    all_messages = initial_team.manager.chat_messages[initial_team.members[0]]
    for message in all_messages:
        _print_received_message(message=message)
