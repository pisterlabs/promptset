from copy import deepcopy
from typing import Any, Dict, List

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult

from aim import Repo
from chatbot_logger import (
    Session, SessionDev, SessionProd,
    MessagesSequence, Message,
    UserActivity, UserActions, UserAction,
)
from aimstack.asp import Metric

"""
An extended Aim callback handler for LangChain.
This callback handler could easily be part of the chatbot_logger.

That would be a more specialized chatbot logger for LangChain.
But we have tried to keep it framework-agnostic.
So it works with other chatbot implementations as well.
These decisions are up to the developers.


About this integration:

There are three main building blocks in Aim logging:
- Objects: a unit of data being saved. Ex. Number, Image, Text etc.
- Sequences: a sequences of objects.
- Containers: a set of interconnected sequences of objects

These constructs enable a unique way of modeling all the relationships in the software being tracked.
Below is an example of how such objects can be tracked and saved for LangChain.

"""


class AimCallbackHandler(BaseCallbackHandler):
    def __init__(
            self,
            username,
            dev_mode,
            experiment,
            repo_path = 'aim://0.0.0.0:53800'
    ) -> None:
        """Initialize callback handler."""

        super().__init__()

        self.repo = Repo.from_path(repo_path)
        self.session = None
        self.messages = None
        self.user_activity = None
        self.user_actions = None
        self.experiment = experiment
        self.username = username
        self.dev_mode = dev_mode

        self.tokens_usage_metric = None
        self.tokens_usage_input = None
        self.tokens_usage_output = None
        self.tokens_usage = None
        self.used_tools = set()

        self.start_inp = None
        self.end_out = None
        self.agent_actions = []

        self.setup()

    def setup(self, **kwargs: Any) -> None:
        if self.session is not None:
            return

        if self.dev_mode:
            self.session = SessionDev(repo=self.repo)
        else:
            self.session = SessionProd(repo=self.repo)

        # System metrics will be tracked by default.
        self.session.enable_system_monitoring()

        # Define what needs to be tracked as a result of LangChain execution
        self.messages = MessagesSequence(self.session, \
            name='messages', context={})

        self.tokens_usage_input = Metric(self.session, \
            name='token-usage-input', context={})

        self.tokens_usage_output = Metric(self.session, \
            name='token-usage-output', context={})

        self.tokens_usage = Metric(self.session, \
            name='token-usage', context={})

        # toy user actions implementation for demo purposes.
        # TODO: Aim api should allow more efficient way of querying the specific user from Aim.
        for cont in self.repo.containers(None, UserActivity):
            if cont['username'] == self.username:
                ua = cont
                for seq in ua.sequences:
                    user_actions = seq
                    break
                else:
                    user_actions = UserActions(ua, name='user-actions', context={})
                break
        else:
            ua = UserActivity(repo=self.repo)
            ua['username'] = self.username
            user_actions = UserActions(ua, name='user-actions', context={})

        self.user_activity = ua
        self.user_actions = user_actions

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts."""
        res = deepcopy(prompts)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        result = deepcopy(response)
        self.tokens_usage_input.track(result.llm_output['token_usage']['prompt_tokens'])
        self.tokens_usage_output.track(result.llm_output['token_usage']['completion_tokens'])
        self.tokens_usage.track(result.llm_output['token_usage']['total_tokens'])

    def on_chain_start(
            self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        inputs_res = deepcopy(inputs)
        self.start_inp = inputs_res['input']

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        outputs_res = deepcopy(outputs)

    def on_tool_start(
            self, serialized: Dict[str, Any], input_str: str, **kwargs: Any
    ) -> None:
        self.agent_actions.append({
            'type': 'tool-start',
            'input': input_str,
        })

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        self.agent_actions.append({
            'type': 'tool-end',
            'input': output,
        })
        self.used_tools.add(kwargs.get('name'))
        self.session.used_tools = list(self.used_tools)


    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> Any:
        self.agent_actions.append({
            'type': 'agent-action',
            'tool': action.tool,
            'tool_input': action.tool_input,
        })

    def on_agent_finish(
            self,
            finish: AgentFinish,
            **kwargs: Any,
    ) -> Any:
        """Run on agent end."""
        self.end_out = finish.return_values['output']
        self.messages.track(Message(self.start_inp, self.end_out, self.agent_actions))
        self.start_inp = None
        self.end_out = None
        self.agent_actions = []
