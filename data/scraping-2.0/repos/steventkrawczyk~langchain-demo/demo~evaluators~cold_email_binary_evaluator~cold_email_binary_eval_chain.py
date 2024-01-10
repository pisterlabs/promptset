"""LLM Chain specifically for evaluating question answering."""
from __future__ import annotations

from typing import Any, List

from langchain import PromptTemplate
from langchain.base_language import BaseLanguageModel
from langchain.chains.llm import LLMChain

from demo.evaluators.cold_email_binary_evaluator.cold_email_binary_eval_prompt import EVAL_PROMPT

class SDREvalChain(LLMChain):
    """LLM Chain specifically for evaluating SDR task performance."""

    @classmethod
    def from_llm(
        cls, llm: BaseLanguageModel, prompt: PromptTemplate = EVAL_PROMPT, **kwargs: Any
    ) -> SDREvalChain:
        """Load SDR Eval Chain from LLM.

        Args:
            llm (BaseLanguageModel): the base language model to use.

            prompt (PromptTemplate): A prompt template containing the input_variables:
            'input', 'answer' and 'result' that will be used as the prompt
            for evaluation.
            Defaults to PROMPT.

            **kwargs: additional keyword arguments.

        Returns:
            QAEvalChain: the loaded QA eval chain.
        """
        expected_input_vars = {"customer", "email"}
        if expected_input_vars != set(prompt.input_variables):
            raise ValueError(
                f"Input variables should be {expected_input_vars}, "
                f"but got {prompt.input_variables}"
            )
        return cls(llm=llm, prompt=prompt, **kwargs)

    def evaluate(
        self,
        examples: List[dict],
        predictions: List[dict],
        customer_key: str = "customer",
        email_key: str = "email",
    ) -> List[dict]:
        """Evaluate question answering examples and predictions."""
        inputs = [
            {
                "customer": example,
                "email": predictions[i],
            }
            for i, example in enumerate(examples)
        ]

        return self.apply(inputs)