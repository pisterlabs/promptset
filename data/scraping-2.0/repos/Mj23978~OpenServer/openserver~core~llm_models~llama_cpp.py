from typing import Any, Dict, List

from .base import BaseLlmModel, LLmInputInterface

from langchain.llms.llamacpp import LlamaCpp
from langchain.schema.output import LLMResult
from langchain.callbacks.base import Callbacks

class LlamaCppModel(BaseLlmModel):
    def __init__(self, inp: LLmInputInterface) -> None:
        LlamaCpp.update_forward_refs()
        if inp.grammer is not None:
            inp.f16_kv=True
        self.client = LlamaCpp(
            model_path=inp.model_name,
            top_k=inp.top_k,
            grammar=inp.grammer,
            grammar_path=inp.grammer_path,
            model_kwargs=inp.model_kwargs,
            top_p=inp.top_p,
            n_ctx=inp.n_ctx,
            f16_kv=inp.f16_kv,
            temperature=inp.temperature,
            n_gpu_layers=inp.n_gpu_layers,
            max_tokens=inp.max_tokens,
            stop=inp.stop,
            cache=inp.cache,
            streaming=inp.stream,
            verbose=True,
            callbacks=inp.callbacks,
        )  # type: ignore

    def compelete(self, prompts: List[str], callbacks: Callbacks = None, metadata: Dict[str, Any] | None = None) -> LLMResult:
        result: LLMResult = self.client.generate(prompts=prompts, metadata=metadata, callbacks=callbacks)
        return result

    async def acompelete(self, prompts: List[str], callbacks: Callbacks = None, metadata: Dict[str, Any] | None = None):
        result = await self.client.agenerate(prompts=prompts, metadata=metadata, callbacks=callbacks)
        return result
