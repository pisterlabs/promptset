from importlib import import_module
from typing import Callable

from langchain.llms.base import LLM
from langchain.schema.embeddings import Embeddings

__all__ = ['load_model']


def load_model(model_type: str, model_info: tuple[str, dict]) -> Embeddings | LLM | None:
	model_name, model_config = model_info
	model_config.pop('template', '')

	module = import_module(f'.{model_name}', 'context_chat_backend.models')

	if module is None or not hasattr(module, 'get_model_for'):
		raise AssertionError(f'Error: could not load {model_name} model')

	get_model_for = module.get_model_for

	if not isinstance(get_model_for, Callable):
		return None

	return get_model_for(model_type, model_config)
