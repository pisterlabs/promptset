"""Create various engines for automata to use."""

from langchain.chat_models import ChatOpenAI
from langchain.llms import BaseLLM

def create_engine(engine: str) -> BaseLLM:
    """Create the model to use."""
    if engine is None:
        return None
    if engine in ["gpt-3.5-turbo", "gpt-4"]:
        return ChatOpenAI(temperature=0, model_name=engine)
    raise ValueError(f"Engine {engine} not supported yet.")
