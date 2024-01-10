"""Run a specific automaton and its sub-automata."""

from functools import partial
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Union

from agent_automata.engines import load_engine
from agent_automata.types import AutomatonRunner, Engine


async def save_text_to_workspace(
    request: str, self_name: str, workspace_name: str
) -> str:
    """Save a file."""
    try:
        input_json = json.loads(request)
        file_name = input_json["file_name"]
        content = input_json["content"]
    except (KeyError, json.JSONDecodeError):
        return "Could not parse input. Please provide the input in the following format: {file_name: <file_name>, description: <description>, content: <content>}"
    path: Path = Path(f"workspace/{workspace_name}/{file_name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(str(content), encoding="utf-8")
    output = f"{self_name}: saved file to `{path.relative_to('workspace')}`"
    print(output)
    return output


async def run_llm_assistant(request: str, engine: Engine) -> str:
    """Run an LLM assistant."""
    from langchain.schema import SystemMessage, HumanMessage

    system_message = SystemMessage(
        content="You are a helpful assistant who can help generate a variety of content. However, if anyone asks you to access files, or refers to something from a past interaction, you will immediately inform them that the task is not possible, and provide no further information."
    )
    request_message = HumanMessage(content=request)
    output = await engine([system_message, request_message])
    print(output)
    return output


def load_builtin_function(
    automaton_id: str,
    automata_location: Path,
    automaton_data: Mapping[str, Any],
    requester_id: str,
) -> AutomatonRunner:
    """Load an automaton function, which are basically wrappers around external functionality (including other agents)."""

    automaton_path = automata_location / automaton_id
    extra_args: Union[None, Mapping[str, Any]] = automaton_data.get("extra_args")

    if automaton_id == "llm_assistant":
        if (
            extra_args is None
            or "engine" not in extra_args
            or extra_args["engine"] is None
        ):
            raise ValueError(
                f'Built-in automaton function `{automaton_id}` requires the "engine" value in the `extra_args` field of the spec.'
            )
        engine_name: str = extra_args["engine"]
        engine: Engine = load_engine(automaton_path, engine_name)  # type: ignore

        return partial(run_llm_assistant, engine=engine)

    elif automaton_id == "save_text":
        run = partial(
            save_text_to_workspace,
            self_name=automaton_data["name"],
            workspace_name=requester_id,
        )

    elif automaton_id == "think":

        async def run(request: str) -> str:
            print(f"Thinking about: {request}")
            return request

    elif automaton_id == "finalize":

        async def run(request: str) -> str:
            print(f"Final Result:\n{request}")
            return request

    else:
        raise NotImplementedError(f"Unsupported function name: {automaton_id}.")

    return run
