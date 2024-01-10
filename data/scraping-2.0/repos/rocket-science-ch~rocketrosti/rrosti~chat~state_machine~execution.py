# Copyright (c) 2023 Rocket Science AG, Switzerland

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""State machine execution."""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from enum import Enum

import marshmallow as ma
from loguru import logger
from overrides import override
from ruamel.yaml import YAML

from rrosti.chat import chat_session
from rrosti.chat.chat_session import Importance, Message
from rrosti.chat.state_machine import ast, interpolable, parsing
from rrosti.llm_api import openai_api, openai_api_direct
from rrosti.query import logging as qlog
from rrosti.servers.websocket_query_server import Frontend, UserInputMessage, WebFrontend
from rrosti.utils import misc
from rrosti.utils.config import config
from rrosti.utils.misc import ProgramArgsBase, ProgramArgsMixinProtocol, truncate_json_strings

yaml = YAML(typ=["rt", "string"])

_NO_PYTHON_CODE_BLOCK_FOUND_MSG = """Your message did not contain a Python code block.

It should look like this:

$$$python
print("Hello world!")
$$$

Please try again. Do not apologize."""

_NO_RTFM_CODE_BLOCK_FOUND_MSG = """Your message did not contain an rtfm code block.

It should look like this:

$$$rtfm
block contents here
$$$

Please try again. Do not apologize."""


_IMPORTANCE_USER_INPUT_INTERPOLABLE = Importance.MEDIUM
_TTL_USER_INPUT_INTERPOLABLE: int | None = None

_IMPORTANCE_PYTHON_INTERPOLABLE = Importance.LOW
_TTL_PYTHON_INTERPOLABLE: int | None = None

_IMPORTANCE_RTFM_INTERPOLABLE = Importance.NOISE
_TTL_RTFM_INTERPOLABLE: int | None = 2


class ProgramArgsMixin(ProgramArgsMixinProtocol):
    user_simulate_llm: bool

    @classmethod
    def _add_args(cls, parser: argparse.ArgumentParser) -> None:
        super()._add_args(parser)
        parser.add_argument(
            "--user-simulate-llm",
            action="store_true",
            help="Instead of using an actual LLM, ask for user input in the console.",
        )


class _Python(interpolable.Interpolable):
    """Implementation of {python()}."""

    _show_output = "Python Output:\n\n"
    _python_vars: dict[str, object]

    def __init__(self, python_vars: dict[str, object]) -> None:
        self._python_vars = python_vars

    @override
    def is_for_me(self, code: str) -> bool:
        return code == "python()"

    # TODO: make async
    @override
    async def execute(self, last_msg: str | None, frontend: Frontend) -> interpolable.InterpolableOutput:
        if last_msg is None:
            logger.error("python() executed without a message in context")
            assert False
        try:
            python_out = interpolable.execute_python_in_msg(last_msg, self._python_vars)
        except interpolable.NoCodeBlockFoundError:
            return interpolable.InterpolableOutput(
                output=_NO_PYTHON_CODE_BLOCK_FOUND_MSG,
                info_prefix="ERROR:\n\n",
                importance=_IMPORTANCE_PYTHON_INTERPOLABLE,
                ttl=_TTL_PYTHON_INTERPOLABLE,
            )
        else:
            frontend.handle_python_output(python_out)
            return interpolable.InterpolableOutput(
                output=python_out._output,
                info_prefix=self._show_output,
                importance=_IMPORTANCE_PYTHON_INTERPOLABLE,
                ttl=_TTL_PYTHON_INTERPOLABLE,
            )


class _Rtfm(interpolable.Interpolable):
    """Implementation of {rtfm()}."""

    _show_output = "RTFM output:\n\n"
    _openai_provider: openai_api.OpenAIApiProvider

    def __init__(self, openai_provider: openai_api.OpenAIApiProvider) -> None:
        self._openai_provider = openai_provider

    @override
    def is_for_me(self, code: str) -> bool:
        return code == "rtfm()"

    @override
    async def execute(self, last_msg: str | None, frontend: Frontend) -> interpolable.InterpolableOutput:
        if last_msg is None:
            logger.error("rtfm() executed without a message in context")
            assert False
        try:
            snippets = await interpolable.execute_rtfm_in_msg(self._openai_provider, last_msg)
        except interpolable.NoCodeBlockFoundError:
            return interpolable.InterpolableOutput(
                output=_NO_RTFM_CODE_BLOCK_FOUND_MSG,
                info_prefix="ERROR:\n\n",
                importance=_IMPORTANCE_RTFM_INTERPOLABLE,
                ttl=_TTL_RTFM_INTERPOLABLE,
            )
        else:
            start_index = frontend.handle_rtfm_output(snippets)
            output_str = "\n-----\n".join(
                f"Extract #{i + start_index}:\n\n{snip.text}" for i, snip in enumerate(snippets)
            ).strip()
            return interpolable.InterpolableOutput(
                output=output_str,
                info_prefix=self._show_output,
                importance=_IMPORTANCE_RTFM_INTERPOLABLE,
                ttl=_TTL_RTFM_INTERPOLABLE,
            )


class _ActionHandler(ast.AsyncActionVisitor):
    runner: _AgentRunner
    config: ast.Config | None

    def __init__(self, runner: _AgentRunner) -> None:
        self.runner = runner
        self.config = None

    @override
    async def message_action(self, action: ast.MessageAction) -> None:
        assert self.config is not None
        logger.info(
            "[Agent {}]: Message action ({}): {}",
            self.runner.name,
            action.role,
            truncate_json_strings(action.text),
        )

        text = action.text

        importances: set[Importance] = set()
        ttls: set[int | None] = set()

        # handle function invocations
        for placeholder, code in action.placeholders.items():
            logger.info("[Agent {}]: Handling function invocation: {}", self.runner.name, code)
            interpolable_output: interpolable.InterpolableOutput
            for interp in self.runner._interpolables:
                if interp.is_for_me(code):
                    interpolable_output = await interp.execute(
                        self.runner._session.messages[-1].text,
                        frontend=self.runner._sm_runner.frontend,
                    )
                    if interpolable_output.info_prefix:
                        await self.runner._sm_runner.frontend.send_message(
                            interpolable_output.info_prefix + interpolable_output.output
                        )
                    break
            else:
                raise ValueError(f"Don't know what to do with interpolation: {code}")

            # We'll probably give the state machine writer the best tools if we strip whitespace
            # from the output, whatever it is.
            replacement = interpolable_output.output.strip()
            text = text.replace(placeholder, replacement)
            importances.add(interpolable_output.importance)
            ttls.add(interpolable_output.ttl)

        if config.state_machine.debug_detect_unresolved_funcalls:
            assert "FUNCALL(" not in text, "Unresolved function call(s)"

        ttl = None
        if not importances:
            # This means there were no function invocations.
            # As a heuristic, we assume that user and system messages added in the initial state
            # are of high importance, and messages added in other states are of medium importance.
            if isinstance(self.runner._curr_state, ast.InitialState):
                importance = Importance.HIGH
            else:
                importance = Importance.MEDIUM
        else:
            # Otherwise, if there's more than one of either, warn and take the min.
            if len(importances) > 1:
                logger.warning(
                    "Multiple importances for message action: {} (taking min)",
                    importances,
                )
            importance = min(importances)
            if len(ttls) > 1:
                logger.warning(
                    "Multiple ttls for message action: {} (taking None)",
                    ttls,
                )
            else:
                ttl = next(iter(ttls))

        self.runner._session.add_message(
            chat_session.Message(role=action.role, text=text, ttl=ttl, importance=importance)
        )

    @override
    async def goto_action(self, action: ast.GotoAction) -> None:
        assert self.config is not None
        logger.info("[Agent {}]: Goto action: {}", self.runner.name, action.label)
        self.runner._curr_state = self.runner._agent.get_state(action.label)

    @override
    async def end_action(self, action: ast.EndAction) -> None:
        assert self.config is not None
        logger.info("[Agent {}]: End action", self.runner.name)
        self.runner._runner_state = RunnerState.TERMINATED

    @override
    async def send_action(self, action: ast.SendAction) -> None:
        assert self.config is not None
        # FIXME: make sure send is the last action.
        logger.info("[Agent {}]: Send action: to: {}; next: {}", self.runner.name, action.to, action.next_state)
        self.runner._runner_state = RunnerState.WAITING

        await self.runner._sm_runner._get_agent_runner(action.to).add_message(
            self.runner._session.messages[-1].as_user_message(), quiet=True
        )


class RunnerState(Enum):
    """
    The state of the agent runner.

    The runner starts in the NOT_STARTED state. The first agent's runner is started by
    StateMachineRunner, putting it in the RUNNING state.

    Apart from this, the only state changes are:
    - Sending a message causes a RUNNING -> WAITING transition.
    - Receiving a message causes a NOT_STARTED/WAITING -> RUNNING transition.
    - Excuting an `end` action causes a RUNNING -> TERMINATED transition.
    """

    NOT_STARTED = "not_started"
    RUNNING = "running"
    WAITING = "waiting"
    TERMINATED = "terminated"


class _AgentRunner:
    """
    The agents are sort-of independent actors that run like coroutines.

    Do not use directly. Use StateMachineRunner.
    """

    _session: chat_session.ChatSession
    _agent: ast.Agent
    _curr_state: ast.State
    _sm_runner: StateMachineRunner
    _inbox: asyncio.Queue[chat_session.Message]
    __runner_state: RunnerState
    _handler: _ActionHandler
    _interpolables: list[interpolable.Interpolable]

    @property
    def _runner_state(self) -> RunnerState:
        return self.__runner_state

    @_runner_state.setter
    def _runner_state(self, value: RunnerState) -> None:
        logger.info("[Agent {}]: State transition: {} -> {}", self.name, self._runner_state, value)
        self.__runner_state = value

    @property
    def _sm(self) -> ast.StateMachine:
        return self._sm_runner._sm

    @property
    def name(self) -> str:
        return self._agent.name

    def __init__(
        self,
        sm_runner: StateMachineRunner,
        agent: ast.Agent,
        llm: chat_session.LLM,
        python_vars: dict[str, object] | None = None,
    ) -> None:
        self._sm_runner = sm_runner
        self._inbox = asyncio.Queue()
        self._agent = agent
        self._session = chat_session.ChatSession(llm, name=self._agent.name, callback=self._sm_runner._message_callback)
        self._curr_state = self._agent.initial_state
        self._handler = _ActionHandler(self)
        self.__runner_state = RunnerState.NOT_STARTED

        async def _get_user_input() -> str:
            # Expire messages with expired ttls.
            self._session.decrease_ttls()
            return (await self._sm_runner.frontend.get_user_input()).content

        self._interpolables = [
            interpolable.SimpleInterpolable(
                code="user_input()",
                importance=_IMPORTANCE_USER_INPUT_INTERPOLABLE,
                ttl=_TTL_USER_INPUT_INTERPOLABLE,
                coro=_get_user_input,
            ),
            _Python(python_vars or {}),
            _Rtfm(self._sm_runner._openai_provider),
        ]

    async def _execute_action(self, action: ast.ActionBase) -> None:
        """Execute an action."""

        assert self._runner_state == RunnerState.RUNNING, self._runner_state
        await action.aaccept(self._handler)

    async def _execute_actions(self, actions: ast.ActionList, config: ast.Config) -> None:
        self._handler.config = config
        await actions.aaccept(self._handler)

    async def start(self) -> None:
        """Start the agent's state machine."""

        assert self._runner_state == RunnerState.NOT_STARTED, self._runner_state
        assert isinstance(self._curr_state, ast.InitialState)
        self._runner_state = RunnerState.RUNNING
        # For the first step, we need to get out of the initial state.
        # For it, we just execute actions. No LLM is invoked here.
        # The initial state is used to set up the environment for the first LLM invocation.
        logger.info("[Agent {}]: Starting.", self._agent.name)
        self._sm_runner._model_override = self._curr_state.config.model
        await self._execute_actions(self._curr_state.action, self._curr_state.config)
        assert isinstance(self._curr_state, ast.NonInitialState), "Did not get out of initial state"
        # self._process_inbox()  # add any messages from inbox

    def _add_message_to_session(self, message: chat_session.Message) -> None:
        """Add a message to the agent's session."""
        assert self._runner_state not in (RunnerState.NOT_STARTED, RunnerState.TERMINATED), self._runner_state
        self._session.add_message(message)
        logger.info("[Agent {}]: Received message: {}", self._agent.name, message)

    def _process_inbox(self) -> None:
        """Process the agent's inbox, adding to the session."""

        if self._inbox.empty():
            return
        assert self._inbox.qsize() == 1, "Sus: Several messages in inbox."
        self._add_message_to_session(self._inbox.get_nowait())

    async def add_message(self, message: chat_session.Message, quiet: bool = False) -> None:
        """Add a message to the agent's inbox."""
        logger.info("[Agent {}]: Adding message to inbox: {}", self._agent.name, message)
        if self._runner_state == RunnerState.NOT_STARTED:
            await self.start()
        elif self._runner_state == RunnerState.WAITING:
            self._runner_state = RunnerState.RUNNING
        self._session.add_message(message, quiet=quiet)

    async def step(self) -> None:
        """Execute one step of the agent's state machine."""

        assert self._runner_state != RunnerState.TERMINATED, "Cannot step after termination"
        if self._runner_state in (RunnerState.NOT_STARTED, RunnerState.WAITING):
            self._add_message_to_session(await self._inbox.get())
        assert self._runner_state == RunnerState.RUNNING, self._runner_state
        assert isinstance(self._curr_state, ast.NonInitialState)
        logger.info("[Agent {}]: Non-initial state: {}", self._agent.name, self._curr_state.name)

        self._process_inbox()  # add any messages from inbox

        msg = await self._session.generate_response(
            self._sm_runner._model_override or self._agent.config.model or self._sm.config.model
        )
        self._sm_runner._model_override = None

        await self._sm_runner.frontend.send_message(msg)

        cond = self._curr_state.triggered_condition(msg.text)

        # If the condition contains a model, it overrides the next model to use
        self._sm_runner._model_override = cond.config.model

        this_state = self._curr_state
        await self._execute_actions(cond.action, cond.config)  # updates self._curr_state

        self._prev_state = this_state

    async def run(self) -> None:
        """Run the agent's state machine until it terminates."""
        assert self._runner_state != RunnerState.TERMINATED, "Cannot run after termination"
        while self._runner_state != RunnerState.TERMINATED:  # type: ignore[comparison-overlap] # (mypy 1.4.1 bug)
            await self.step()

    def __repr__(self) -> str:
        return f"<_AgentRunner {self._agent.name}>"


class StateMachineRunner:
    """Execute a network of agents (essentially a hierarchical state machine)."""

    _sm: ast.StateMachine
    _llm: chat_session.LLM
    _python_vars: dict[str, object]
    _agent_runners: list[_AgentRunner]
    frontend: Frontend
    _message_callback: chat_session.MessageCallback | None
    _openai_provider: openai_api.OpenAIApiProvider

    _model_override: str | None = None
    """When an action specifies a model, it is stored here and used for the next message."""

    @property
    def _is_terminated(self) -> bool:
        return any(runner._runner_state == RunnerState.TERMINATED for runner in self._agent_runners)

    @property
    def _running_agent(self) -> _AgentRunner:
        assert not self._is_terminated, "Cannot get running agents after termination"
        rs = [runner for runner in self._agent_runners if runner._runner_state == RunnerState.RUNNING]
        assert len(rs) == 1, f"Exactly one agent should be running, got {len(rs)}: {rs}"
        return rs[0]

    def __init__(
        self,
        *,
        sm: ast.StateMachine,
        llm: chat_session.LLM,
        frontend: Frontend,
        openai_provider: openai_api.OpenAIApiProvider,
        python_vars: dict[str, object] | None = None,
        message_callback: chat_session.MessageCallback | None = None,
    ) -> None:
        """
        Args:
            sm: The state machine to execute.
            llm: The LLM to use.
            frontend: an Object that defines and handles communication with the Frontend
            openai_provider: The OpenAI provider to use.
            python_vars: The Python variables to use.
            message_callback: A callback to call when a message is sent or received.
        """
        self._openai_provider = openai_provider
        self._sm = sm
        self._llm = llm
        self._python_vars = python_vars or {}
        self._message_callback = message_callback
        self._agent_runners = [_AgentRunner(self, agent, self._llm, self._python_vars) for agent in sm.agents]
        self.frontend = frontend

    def _get_agent_runner(self, name: str) -> _AgentRunner:
        """Get the agent runner with the given name."""
        for runner in self._agent_runners:
            if runner.name == name:
                return runner
        raise ValueError(f"Agent {name} not found")

    async def step(self) -> None:
        """Execute one step of the state machine."""

        # The first agent is the one that starts the state machine.
        if self._agent_runners[0]._runner_state == RunnerState.NOT_STARTED:
            await self._agent_runners[0].start()

        assert not self._is_terminated, "Cannot step after termination"
        await self._running_agent.step()

    async def run(self) -> None:
        """Run the state machine until it terminates."""
        assert not self._is_terminated, "Cannot run after termination"
        while not self._is_terminated:
            await self.step()


class MessageObserver:
    total_cost = 0.0

    def __call__(self, message: chat_session.Message, agent: str | None, quiet: bool) -> None:
        if not quiet:
            print(message.to_string(agent=agent))
        if message.cost:
            self.total_cost += message.cost
            logger.info("Total cost so far: {:.5f}", self.total_cost)


@misc.async_once_in_thread
def _load_state_machine() -> tuple[str, ast.StateMachine]:
    yaml_data = config.state_machine.yaml_path.read_text()
    sm = parsing.loads_from_yaml(yaml_data)
    yaml.indent(mapping=4, sequence=4, offset=0)
    logger.info(
        "Loaded state machine:\n{}",
        yaml.dump_to_string(truncate_json_strings(sm.to_json())),  # type: ignore[attr-defined]
    )
    return yaml_data, sm


async def ensure_loaded() -> None:
    await _load_state_machine()


async def load_and_run(
    *,
    openai_provider: openai_api.OpenAIApiProvider,
    llm: chat_session.LLM,
    frontend: Frontend,
) -> None:
    yaml_text, sm = await _load_state_machine()

    # Get the first user input as the query, just for logging purposes.
    query = await frontend.peek_user_input()
    logger.info("User conversation starter: {}", query)

    with qlog.QueryEvent.section(text=query.content, uuid=query.uuid, prompt=yaml_text):
        runner = StateMachineRunner(
            sm=sm,
            llm=llm,
            python_vars=dict(),
            message_callback=MessageObserver(),
            frontend=frontend,
            openai_provider=openai_provider,
        )

        await runner.run()


class ProgramArgs(ProgramArgsBase, ProgramArgsMixin):
    @classmethod
    def _add_args(cls, parser: argparse.ArgumentParser) -> None:
        super()._add_args(parser)


class FakeWebFrontend(WebFrontend):
    def __init__(self) -> None:
        super().__init__(None, False)

    @override
    async def send_message(self, msg: Message | str) -> None:
        # TODO: implement console version, if needed
        pass

    @override
    async def _get_user_input_impl(self) -> UserInputMessage:
        # TODO: implement console version, if needed
        return UserInputMessage(content="", uuid=str(uuid.uuid4()))


async def main() -> None:
    misc.setup_logging()
    openai_provider = openai_api_direct.DirectOpenAIApiProvider()
    args = ProgramArgs.parse_args()
    llm: chat_session.LLM
    if args.user_simulate_llm:
        llm = chat_session.UserInputLLM()
    else:
        llm = chat_session.OpenAI(openai_provider)
    try:
        await load_and_run(llm=llm, frontend=FakeWebFrontend(), openai_provider=openai_provider)
    except ma.exceptions.ValidationError as e:
        print(e.messages, sys.stderr)
        raise


if __name__ == "__main__":
    asyncio.run(main())
