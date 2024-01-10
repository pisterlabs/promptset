from abc import ABC, abstractmethod
from functools import partial
from cpoison.eval import EvalTuple
from typing import Any, Callable, Coroutine, Literal
from cpoison.eval import EvalTuple, evaluate_model
from lowstakes.llm import OpenAIChatModel, gpt_3_5, gpt_4, call_llm, Role
from cpoison.config import TRUSTED_MODEL, UNTRUSTED_MODEL
from lowstakes.utils import AndMetadata, Metadata
from cpoison.base_models import DirectModel, Model


Annotator = Callable[[EvalTuple], Coroutine[Any, Any, AndMetadata[bool]]]


def get_simple_annotator(model: Model) -> Annotator:
    return partial(evaluate_model, model=model)
