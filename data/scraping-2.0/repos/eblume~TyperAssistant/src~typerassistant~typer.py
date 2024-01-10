from __future__ import annotations

import sys
from dataclasses import KW_ONLY, dataclass, field
from typing import Callable, Iterable, Optional, Type

import typer
from openai import OpenAI
from typer.main import get_command_from_info

from .assistant import Assistant, AssistantT
from .spec import FunctionSpec, ParameterSpec


@dataclass
class TyperAssistant(Assistant):
    """An Assistant generated from a Typer app."""

    app: typer.Typer
    _: KW_ONLY
    instructions: str = "The agent is an interface to a python Typer CLI. The tools available correspond to typer commands. Please help the user with their queries, executing CLI functions as needed. Be concise, but don't shorten the function names even if they look like file paths."
    name: str = field(init=False)

    def __post_init__(self):
        # In AppAssistant, we always infer the name
        self.name = self.app.info.name or sys.argv[0]

    @classmethod
    def from_id(cls: Type[AssistantT], assistant_id: str, client: Optional[OpenAI] = None) -> AssistantT:
        """from_id is disabled for TyperAssistant, use from_id_with_app instead."""
        # TODO this is ugly and bad but I could not find a way to satisfy LSP.
        # Even uglier: to get the unused variable warning to go away, these two asserts:
        assert client or True
        assert assistant_id or True
        # Just an ugly, ugly function. Two out of five stars. I'm sorry.
        raise NotImplementedError("from_id is disabled for TyperAssistant, use from_id_with_app instead.")

    @classmethod
    def from_id_with_app(cls, assistant_id: str, app: typer.Typer, client: Optional[OpenAI] = None) -> TyperAssistant:
        if client is None:
            client = OpenAI()
        assistant = client.beta.assistants.retrieve(assistant_id)
        return TyperAssistant(
            app=app, client=client, instructions=assistant.instructions or cls.instructions, _assistant=assistant
        )

    def functions(self) -> Iterable[FunctionSpec]:
        """Generate FunctionSpecs from the Typer app."""
        yield from super().functions()  # currently a non-op but may be useful to others
        for func in typerfunc(self.app):
            yield func


def register_assistant(
    app: typer.Typer,
    command_name: str = "ask",
    client: Optional[OpenAI] = None,
    client_factory: Optional[Callable[[], OpenAI]] = None,
) -> None:
    """Create a command for the typer application that queries an automatically generated assistant."""
    if client is not None and client_factory is not None:
        raise ValueError("Cannot specify both client and client_factory")

    if client_factory is not None:
        client = client_factory()
    elif client is None:
        client = OpenAI()

    def _ask_command(
        query: str, use_commands: bool = True, confirm_commands: bool = False, replace_assistant: bool = False
    ):
        """Ask an assistant for help, optionally using other commands from this application."""
        assistant = TyperAssistant(app=app, replace=replace_assistant)
        print(assistant.ask(query, use_commands=use_commands, confirm_commands=confirm_commands))

    app.command(command_name, context_settings={"obj": {"omit_from_assistant": True}})(_ask_command)


def typerfunc(app: typer.Typer, command_prefix: Optional[str] = None) -> list[FunctionSpec]:
    """Returns a list of FunctionSpecs describing the CLI of app.

    This function recurses on command groups, with a command_prefix appended to the beginning of each command name in
    that group.

    Omits commands with context_settings['omit_from_assistant'] set to True.
    """
    if command_prefix is None:
        if isinstance(app.info.name, str):
            command_prefix = app.info.name
        else:
            command_prefix = sys.argv[0]

    functions: list[FunctionSpec] = []

    for command_info in app.registered_commands or []:
        if command_info.context_settings and command_info.context_settings.get("omit_from_assistant", False):
            continue

        command = get_command_from_info(
            command_info=command_info,
            pretty_exceptions_short=app.pretty_exceptions_short,
            rich_markup_mode=app.rich_markup_mode,
        )
        assert command.name is not None
        # I'm not sure where it happens, but it's documented here: https://typer.tiangolo.com/tutorial/commands/name/
        #     "Note that any underscores in the function name will be replaced with dashes."
        # Therefore, convert all dashes back to underscores. *shrug*
        fullname = f"{command_prefix}.{command.name.replace('-', '_')}"

        # Extract callback signature for parameters.
        params = []
        for param in command.params:
            descr = getattr(param, "help", "None")
            assert param.name is not None

            param_spec = ParameterSpec(
                name=param.name,
                description=descr,
                default=str(param.default) if param.default is not None else None,
                required=param.required,
            )

            params.append(param_spec)

        assert command_info.callback is not None
        spec = FunctionSpec(
            name=fullname,
            description=getattr(command, "help", "None"),
            parameters=params,
            action=command_info.callback,
        )
        functions.append(spec)

    # Iterate over registered groups, recursing on each
    for group in app.registered_groups:
        # As with the command name, convert all dashes to underscores.
        assert group.typer_instance is not None
        assert group.name is not None
        functions.extend(
            typerfunc(group.typer_instance, command_prefix=command_prefix + "." + group.name.replace("-", "_"))
        )

    return functions
