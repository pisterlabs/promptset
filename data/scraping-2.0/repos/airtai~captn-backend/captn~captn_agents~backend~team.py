import json
from typing import Any, Callable, Dict, List, Optional

import autogen
import openai
from fastcore.basics import patch

from .config import CONFIG_LIST

_completions_create_original = autogen.oai.client.OpenAIWrapper._completions_create


# WORKAROUND for consistent 500 eror code when using openai functions
@patch  # type: ignore
def _completions_create(
    self: autogen.oai.client.OpenAIWrapper,
    client: openai.OpenAI,
    params: Dict[str, Any],
) -> Any:
    for message in params["messages"]:
        name = message.get("name")
        role = message.get("role")
        if name and role != "function":
            # print(f"Removing name parameter from the following message:\n{message}")
            message.pop("name")

    tokens_per_request = autogen.token_count_utils.count_token(
        params["messages"], model="gpt-4-1106-preview"
    )
    print(f"Tokens per request: {tokens_per_request}")

    return _completions_create_original(self, client=client, params=params)


class Team:
    _team_name_counter: int = 0
    _functions: Optional[List[Dict[str, Any]]] = None

    _teams: Dict[str, "Team"] = {}

    @staticmethod
    def _store_team(team_name: str, team: "Team") -> None:
        if team_name in Team._teams:
            raise ValueError(f"Team name '{team_name}' already exists")

        Team._teams[team_name] = team

    @staticmethod
    def get_team(team_name: str) -> "Team":
        try:
            return Team._teams[team_name]
        except KeyError as e:
            raise ValueError(f"Unknown team name: '{team_name}'") from e

    @staticmethod
    def get_user_conv_team_name(name_prefix: str, user_id: int, conv_id: int) -> str:
        name = f"{name_prefix}_{str(user_id)}_{str(conv_id)}"

        return name

    def __init__(
        self,
        roles: List[Dict[str, str]],
        name: str,
        function_map: Optional[Dict[str, Callable[[Any], Any]]] = None,
        work_dir: str = "my_default_workdir",
        max_round: int = 80,
        seed: int = 42,
        temperature: float = 0.2,
        human_input_mode: str = "NEVER",
    ):
        self.roles = roles
        self.initial_message: str
        self.name: str
        self.max_round = max_round
        self.seed = seed
        self.temperature = temperature

        self.function_map = function_map
        self.work_dir = work_dir
        self.llm_config: Optional[Dict[str, Any]] = None
        self.name = name
        self.human_input_mode = human_input_mode
        Team._store_team(self.name, self)

    @classmethod
    def _get_new_team_name(cls) -> str:
        name_prefix = cls._get_team_name_prefix()
        name = f"{name_prefix}_{cls._team_name_counter}"
        cls._team_name_counter = cls._team_name_counter + 1

        return name

    @classmethod
    def _get_team_name_prefix(cls) -> str:
        raise NotImplementedError()

    @classmethod
    def get_llm_config(cls, seed: int = 42, temperature: float = 0.2) -> Dict[str, Any]:
        llm_config = {
            "config_list": CONFIG_LIST,
            "seed": seed,
            "temperature": temperature,
            "functions": cls._functions,
            # "request_timeout": 800,
        }
        return llm_config

    def _create_groupchat_and_manager(self) -> None:
        manager_llm_config = self.llm_config.copy()  # type: ignore
        # GroupChatManager is not allowed to make function/tool calls (from version 0.2.2).
        manager_llm_config.pop("functions", None)
        manager_llm_config.pop("tools", None)

        self.groupchat = autogen.GroupChat(
            agents=self.members, messages=[], max_round=self.max_round
        )
        self.manager = autogen.GroupChatManager(
            groupchat=self.groupchat,
            llm_config=manager_llm_config,
            is_termination_msg=self._is_termination_msg,
        )

    def _create_members(self) -> None:
        self.members = [
            self._create_member(role["Name"], role["Description"])
            for role in self.roles
        ]
        self._create_groupchat_and_manager()

    @staticmethod
    def _is_termination_msg(x: Dict[str, Optional[str]]) -> bool:
        content = x.get("content")
        if content is None:
            return False

        content_xs = content.split()
        return len(content_xs) > 0 and (
            "TERMINATE" in content_xs[-1] or "PAUSE" in content_xs[-1]
        )

    def _create_member(
        self,
        name: str,
        description: str,
        is_user_proxy: bool = False,
    ) -> autogen.ConversableAgent:
        name = name.lower().replace(" ", "_")
        system_message = f"""You are {name}, {description}

Your task is to chat with other team mambers and try to solve the given task.
Do NOT try to finish the task until other team members give their opinion.
"""

        if is_user_proxy:
            return autogen.UserProxyAgent(
                human_input_mode=self.human_input_mode,
                name=name,
                llm_config=self.llm_config,
                system_message=system_message,
                is_termination_msg=self._is_termination_msg,
            )

        return autogen.AssistantAgent(
            name=name,
            llm_config=self.llm_config,
            system_message=system_message,
            is_termination_msg=self._is_termination_msg,
            code_execution_config={"work_dir": self.work_dir},
            function_map=self.function_map,
        )

    @property
    def _first_section(self) -> str:
        roles = ", ".join([str(member.name) for member in self.members])
        return f"""The team is consisting of the following roles: {roles}.


Play to your strengths as an LLM and pursue simple strategies with no legal complications.
"""

    @property
    def _task(self) -> str:
        raise NotImplementedError()

    @property
    def _guidelines(self) -> str:
        raise NotImplementedError()

    @property
    def _constraints(self) -> str:
        return """## Constraints
You operate within the following constraints:
1. ~4000 word limit for short term memory. Your short term memory is short, so immediately save important information to files.
2. If you are unsure how you previously did something or want to recall past events, thinking about similar events will help you remember.
3. You can ask and answer questions from other team members or suggest function listed below e.g. command_name
4. The context size is limited so try to be as concise in discussinos as possible. Do not reapeat yourself or others
"""

    @property
    def _commands(self) -> str:
        raise NotImplementedError()

    @property
    def _resources(self) -> str:
        return """## Resources
You can leverage access to the following resources:
1. Long Term memory management.
2. File output.
3. Command execution
"""

    @property
    def _best_practices(self) -> str:
        return """## Best practices
1. Continuously review and analyze your actions to ensure you are performing to the best of your abilities.
2. Constructively self-criticize your big-picture behavior constantly.
3. Reflect on past decisions and strategies to refine your approach.
4. Every command has a cost, so be smart and efficient. Aim to complete tasks in the least number of steps.
5. If you have some doubts, ask question.
"""

    @property
    def _final_section(self) -> str:
        return ""

    def _create_initial_message(self) -> None:
        self.initial_message = f"""{self._task}

{self._first_section}

{self._guidelines}

{self._constraints}

{self._commands}

{self._resources}

{self._best_practices}

{self._final_section}
"""

    def initiate_chat(self) -> None:
        self.manager.initiate_chat(self.manager, message=self.initial_message)

    async def a_initiate_chat(self) -> None:
        await self.manager.a_initiate_chat(self.manager, message=self.initial_message)

    def get_last_message(self, add_prefix: bool = True) -> str:
        last_message = self.groupchat.messages[-1]["content"]
        if isinstance(last_message, dict):
            last_message = json.dumps(last_message)
        last_message = last_message.replace("PAUSE", "").replace("TERMINATE", "")

        if add_prefix:
            last_message = f"Response from team '{self.name}':\n{last_message}"

        return last_message  # type: ignore

    def continue_chat(self, message: str) -> None:
        message = f"Response from the client:\n{message}"
        self.manager.send(recipient=self.manager, message=message)

    async def a_continue_chat(self, message: str) -> None:
        message = f"Response from the client:\n{message}"
        await self.manager.a_send(recipient=self.manager, message=message)
