from typing import List

from langchain.tools import APIOperation
from pydantic.main import BaseModel
from typing_extensions import override

from aidial_assistant.application.prompts import (
    ADDON_BEST_EFFORT_TEMPLATE,
    ADDON_SYSTEM_DIALOG_MESSAGE,
)
from aidial_assistant.chain.command_chain import (
    CommandChain,
    CommandConstructor,
)
from aidial_assistant.chain.history import History, ScopedMessage
from aidial_assistant.commands.base import (
    Command,
    ExecutionCallback,
    ResultObject,
    TextResult,
)
from aidial_assistant.commands.open_api import OpenAPIChatCommand
from aidial_assistant.commands.plugin_callback import PluginChainCallback
from aidial_assistant.commands.reply import Reply
from aidial_assistant.model.model_client import (
    Message,
    ModelClient,
    ReasonLengthException,
)
from aidial_assistant.open_api.operation_selector import collect_operations
from aidial_assistant.utils.open_ai_plugin import OpenAIPluginInfo


class PluginInfo(BaseModel):
    info: OpenAIPluginInfo
    auth: str | None


class RunPlugin(Command):
    def __init__(
        self,
        model_client: ModelClient,
        plugins: dict[str, PluginInfo],
        max_completion_tokens: int,
    ):
        self.model_client = model_client
        self.plugins = plugins
        self.max_completion_tokens = max_completion_tokens

    @staticmethod
    def token():
        return "run-addon"

    @override
    async def execute(
        self, args: List[str], execution_callback: ExecutionCallback
    ) -> ResultObject:
        self.assert_arg_count(args, 2)
        name = args[0]
        query = args[1]

        return await self._run_plugin(name, query, execution_callback)

    async def _run_plugin(
        self, name: str, query: str, execution_callback: ExecutionCallback
    ) -> ResultObject:
        if name not in self.plugins:
            raise ValueError(
                f"Unknown addon: {name}. Available addons: {[*self.plugins.keys()]}"
            )

        plugin = self.plugins[name]
        info = plugin.info
        ops = collect_operations(info.open_api, info.ai_plugin.api.url)
        api_schema = "\n\n".join([op.to_typescript() for op in ops.values()])  # type: ignore

        def create_command(op: APIOperation):
            return lambda: OpenAPIChatCommand(op, plugin.auth)

        command_dict: dict[str, CommandConstructor] = {}
        for name, op in ops.items():
            # The function is necessary to capture the current value of op.
            # Otherwise, only first op will be used for all commands
            command_dict[name] = create_command(op)

        command_dict[Reply.token()] = Reply

        history = History(
            assistant_system_message_template=ADDON_SYSTEM_DIALOG_MESSAGE.build(
                command_names=ops.keys(),
                api_description=info.ai_plugin.description_for_model,
                api_schema=api_schema,
            ),
            best_effort_template=ADDON_BEST_EFFORT_TEMPLATE.build(
                api_schema=api_schema
            ),
            scoped_messages=[ScopedMessage(message=Message.user(query))],
        )

        chat = CommandChain(
            model_client=self.model_client,
            name="PLUGIN:" + name,
            command_dict=command_dict,
            max_completion_tokens=self.max_completion_tokens,
        )

        callback = PluginChainCallback(execution_callback)
        try:
            await chat.run_chat(history, callback)
            return TextResult(callback.result)
        except ReasonLengthException:
            return TextResult(callback.result)
