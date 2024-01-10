from typing import Dict, Optional, Any

from langchain.callbacks.manager import AsyncCallbackManagerForChainRun
from langchain.chains import LLMChain
from langchain.llms.base import LLM
from langchain.schema import BasePromptTemplate
from pydantic import Extra, parse_obj_as

from src.packages.chains.api_prompted_chain import APIPromptedChain


class ConversationalAPIChain(LLMChain):
    prompt: BasePromptTemplate
    """Prompt object to use."""
    llm: LLM
    output_key: str = "answer"

    # region Embedded Chains

    api_chain: APIPromptedChain
    classification_chain: LLMChain

    # endregion

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        response: dict[str, str] = self.classification_chain(inputs=inputs)
        answer: str = response.get(self.classification_chain.output_key, 'False').strip().lower()
        is_conversational_query: bool = parse_obj_as(bool, answer)

        if not is_conversational_query:
            response: dict[str, str] = self.api_chain(inputs=inputs)

            return response

        # Proceed with conversational chain

        prompt_value = self.prompt.format_prompt(**{
            key: value for key, value in inputs.items()
            if key in self.prompt.input_variables
        })

        response = await self.llm.agenerate_prompt(
            [prompt_value], callbacks=run_manager.get_child() if run_manager else None
        )

        return {self.output_key: response.generations[0][0].text}

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def _chain_type(self) -> str:
        return "conversational_api_chain"
