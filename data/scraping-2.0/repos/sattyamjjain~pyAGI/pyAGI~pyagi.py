from typing import List, Dict, Any
from langchain.chains.base import Chain
from pydantic import BaseModel

from pyAGI.utils import setup_logger, log_header, log_content
from pyAGI.prompts import (
    get_architecture_generator_prompt,
    get_ux_flow_generator_prompt,
    get_code_flow_generator_prompt,
    get_coding_steps_generator_prompt,
    get_app_code_generator_prompt,
    get_description_generator_prompt,
)
from pyAGI.pyAGIChainGenerator import GeneratePyAGIChain

logger = setup_logger(__name__)


class PyAGI(Chain, BaseModel):
    llm_chain: GeneratePyAGIChain

    class Config:
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        return ["objective", "selected_model"]

    @property
    def output_keys(self) -> List[str]:
        return []

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        objective = inputs["objective"]
        selected_model = inputs["selected_model"]
        log_header(logger, "OBJECTIVE AREA")
        log_content(logger, objective)

        (
            desc_outcome,
            desc_prompt,
            desc_prompt_max_token,
        ) = get_description_generator_prompt()
        if not self.llm_chain.is_chat_model(selected_model):
            self.llm_chain.llm.max_tokens = desc_prompt_max_token
        description = self.llm_chain.run(
            objective=objective, maincontent=desc_prompt, outcome=desc_outcome
        )
        log_content(logger, description)

        (
            arc_outcome,
            arc_prompt,
            arc_prompt_max_token,
        ) = get_architecture_generator_prompt(description)
        if not self.llm_chain.is_chat_model(selected_model):
            self.llm_chain.llm.max_tokens = arc_prompt_max_token
        architecture = self.llm_chain.run(
            objective=objective,
            maincontent=arc_prompt,
            outcome=arc_outcome,
        )
        log_content(logger, architecture)

        ux_outcome, ux_prompt, ux_prompt_max_token = get_ux_flow_generator_prompt(
            description, architecture
        )
        if not self.llm_chain.is_chat_model(selected_model):
            self.llm_chain.llm.max_tokens = ux_prompt_max_token
        uxFlow = self.llm_chain.run(
            objective=objective, maincontent=ux_prompt, outcome=ux_outcome
        )
        log_content(logger, uxFlow)

        (
            code_flow_outcome,
            code_flow_prompt,
            code_flow_prompt_max_token,
        ) = get_code_flow_generator_prompt(description, architecture, uxFlow)
        if not self.llm_chain.is_chat_model(selected_model):
            self.llm_chain.llm.max_tokens = code_flow_prompt_max_token
        codeflow = self.llm_chain.run(
            objective=objective, maincontent=code_flow_prompt, outcome=code_flow_outcome
        )
        log_content(logger, codeflow)

        (
            coding_steps_outcome,
            coding_steps_prompt,
            coding_steps_prompt_max_token,
        ) = get_coding_steps_generator_prompt(
            description, architecture, uxFlow, codeflow
        )
        if not self.llm_chain.is_chat_model(selected_model):
            self.llm_chain.llm.max_tokens = coding_steps_prompt_max_token
        codingSteps = self.llm_chain.run(
            objective=objective,
            maincontent=coding_steps_prompt,
            outcome=coding_steps_outcome,
        )
        log_content(logger, codingSteps)

        (
            app_code_outcome,
            app_code_prompt,
            app_code_max_token,
        ) = get_app_code_generator_prompt(
            description, architecture, uxFlow, codeflow, codingSteps
        )
        if not self.llm_chain.is_chat_model(selected_model):
            self.llm_chain.llm.max_tokens = app_code_max_token
        appcode = self.llm_chain.run(
            objective=objective, maincontent=app_code_prompt, outcome=app_code_outcome
        )
        log_content(logger, appcode)
        log_header(logger, "THANK YOU (from pyAGI)")

        return {}

    @classmethod
    def create_llm_chain(
        cls, verbose: bool = False, selected_model: str = None
    ) -> "PyAGI":
        llm_chain = GeneratePyAGIChain.create_chain(
            verbose=verbose, selected_model=selected_model
        )
        return cls(llm_chain=llm_chain)
