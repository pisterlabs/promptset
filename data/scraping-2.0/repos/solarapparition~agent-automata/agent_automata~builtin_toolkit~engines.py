"""Builtin LLM engines that can be used by automata."""

from typing import Any, Sequence, Union

from agent_automata.types import Engine

BUILTIN_ENGINES = {"gpt-3.5-turbo", "gpt-4"}


def load_builtin_engine(name: str) -> Engine:
    """Load a builtin engine."""

    if name in ["gpt-3.5-turbo", "gpt-4"]:
        from langchain.chat_models import ChatOpenAI

        model = ChatOpenAI(temperature=0, model_name=name, verbose=True)

        async def run_model(prompt: Union[str, Sequence[Any]], **kwargs: Any) -> str:
            if isinstance(prompt, str):
                return await model.apredict(prompt, **kwargs)
            return (await model.apredict_messages(prompt, **kwargs)).content

        return run_model

    raise ValueError(f"Engine {name} not part of builtin engines: `{BUILTIN_ENGINES}`")
