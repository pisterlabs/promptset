from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

from personalizer_prompt import PROMPT
from langchain.prompts.prompt import PromptTemplate

from pydantic import Extra, PrivateAttr

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain

class GraphFeedbackChain(Chain):
    
    llm_chains: Dict[str, LLMChain]
    prompt: PromptTemplate
    llm_name: str = "selected"  #: :meta private:
    output_key: str = "result"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @property
    def input_keys(self) -> List[str]:
        """Expect input key.

        :meta private:
        """
        return [self.llm_name]

    @property
    def output_keys(self) -> List[str]:
        """Expect output key.

        :meta private:
        """
        return [self.output_key]

    def _call(self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun] = None,) -> Dict[str, Any]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        
        llm = inputs[self.llm_name]

        if llm not in self.llm_chains.keys():
            raise ValueError("selected must be in llm_chains passed during construction of the chain")
        
        llm_chain = self.llm_chains[llm]
        t = llm_chain.predict(
            **inputs,
            callbacks=_run_manager.get_child(),
        )

        output = t.strip()
        return {self.output_key: output}

    @classmethod
    def from_llms(
        cls, llms: Dict[str, BaseLanguageModel], prompt: PromptTemplate = PROMPT, **kwargs: Any
    ) -> GraphFeedbackChain:
        llm_chains = {name: LLMChain(llm=llm, prompt=prompt) for name, llm in llms.items()}
        return GraphFeedbackChain.from_llm_chains(llm_chains=llm_chains, prompt=prompt, **kwargs)
    
    @classmethod
    def from_llm_chains(
        cls, llm_chains: Dict[str, LLMChain], prompt: PromptTemplate = PROMPT, **kwargs: Any
    ) -> GraphFeedbackChain:
            return GraphFeedbackChain(llm_chains=llm_chains, prompt=prompt, **kwargs)