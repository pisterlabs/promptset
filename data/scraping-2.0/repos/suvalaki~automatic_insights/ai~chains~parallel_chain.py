from typing import Dict, Any, Optional, List, Callable, Tuple, Union
from pydantic import root_validator, BaseModel

from langchain import PromptTemplate, LLMChain
from langchain.chains.base import Chain
from langchain.chains.transform import TransformChain
from langchain.llms.base import BaseLanguageModel
from langchain.callbacks.manager import (
    Callbacks,
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)


class ExtractChain(TransformChain):
    transform: Callable[[Dict[str, str]], List[Dict[str, str]]]
    output_variables: List[str] = ["extracted"]

    @property
    def output_key(self) -> str:
        return self.output_variables[0]


class ParallelChain(Chain):
    chain: Chain  # A stateless chain
    extract_inputs: ExtractChain  # Get additional data if required. eg db enrichment Transformer
    input_prepper: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None

    def prep_inputs(self, inputs: Union[Dict[str, Any], Any]) -> Dict[str, str]:
        inputs = super().prep_inputs(inputs)
        if self.input_prepper:
            inputs = self.input_prepper(inputs)
        return inputs

    # aprep

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        # Gets a list / generator to roll across
        extract_inputs = self.extract_inputs(inputs, run_manager)

        input_filter = lambda inputs, extracted: {
            k: v
            for k, v in inputs.items()
            if k not in extracted.keys() and k in self.chain.input_keys
        }

        if getattr(self.chain, "predict", None):
            replies = [
                self.chain.predict(
                    **extracted,
                    **input_filter(inputs, extracted),
                    callbacks=run_manager
                )
                for extracted in extract_inputs[self.extract_inputs.output_key]
            ]
        else:
            replies = [
                self.chain(
                    {**extracted, **input_filter(inputs, extracted)},
                    callbacks=run_manager,
                )
                for extracted in extract_inputs[self.extract_inputs.output_key]
            ]

        return {self.output_key: replies}

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        # Gets a list / generator to roll across
        extract_inputs = await self.extract_inputs(inputs, run_manager)
        replies = [
            self.chain.apredict(**extracted, run_manager=run_manager)
            async for extracted in extract_inputs[self.extract_inputs.output_key]
        ]

        return {self.output_key: replies}

    def predict(self, callbacks: Callbacks = None, **kwargs: Any) -> List[Any]:
        return self(kwargs, callbacks=callbacks)[self.output_key]

    async def apredict(self, callbacks: Callbacks = None, **kwargs: Any) -> List[Any]:
        return (await self.acall(kwargs, callbacks=callbacks))[self.output_key]

    @property
    def _chain_type(self) -> str:
        return (
            "parallel_" + self.chain._chain_type + "_" + self.extract_inputs._chain_type
        )
