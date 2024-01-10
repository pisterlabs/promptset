from __future__ import annotations
import os
import fitz
import json
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
from pydantic import Extra, root_validator, Field, BaseModel, validator
from loguru import logger

from langchain.schema.language_model import BaseLanguageModel
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate, BasePromptTemplate
from langchain.chains.base import Chain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)

from meche_copilot.schemas import ScopedEquipment, ScopedEquipmentInstance, EquipmentSpecificationAnalysis
from meche_copilot.utils.converters import pydantic_from_jsonl, pydantic_to_jsonl, title_to_filename
from meche_copilot.utils.envars import OPENAI_API_KEY, DATA_CACHE

# TODO - everywhere a llm/prompt is used, it should include a way to chunk the data in case it is too long for the model

# class for llm output parsing
class SpecificationResults(BaseModel):
    eq_uid: str = Field(None, title="Equipment Instance UID")
    spec_name: str = Field(None, title="Specification Name")
    design_result: str = Field(None, title="The specification result from the design data, if appropriate or None")
    submittal_result: str = Field(None, title="The specification result from the submittal data, if appropriate or None")
    confidence: float = Field(None, title="Confidence in your result")
    notes: str = Field(None, title="Notes about the result or questions you have if you are unsure")

class SpecificationAnalysis(BaseModel):
    eq_uid: str = Field(None, title="Equipment Instance UID")
    spec_name: str = Field(None, title="Specification Name")
    final_result: str = Field(None, title="The final specification result")
    design_notes: str = Field(None, title="Notes about the design result")
    submittal_notes: str = Field(None, title="Notes about the submittal result")
    confidence: float = Field(None, title="Confidence in your result")
    notes: str = Field(None, title="Notes about the result or questions you have if you are unsure")
class AnalyzeSpecsChain(Chain):
    
    prompt: BasePromptTemplate = PromptTemplate.from_template('') # TODO - use build extras?
    chat = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model="gpt-4")
    output_key: str = "result" #: :meta private:

    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        return self.prompt.input_keys
    
    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        """Run the chain.
        """

        logger.debug(f"input keys: {inputs.keys()}")
        scoped_eq: List[ScopedEquipment] = inputs.get('scoped_eq', [])
        refresh_submittal_data: bool = inputs.get('refresh_submittal_data', False)
        kwargs = getattr(self, 'kwargs', {})

        show_your_work: bool = kwargs.get('show_your_work', False)
        if show_your_work:
            logger.add(sink=self.submittal_data_cache / 'logs/read_design_chain.log', rotation="1 week", level="DEBUG")

        if refresh_submittal_data: # remove cached
            logger.info(f"Removing cached design data...")
            os.remove(str(self.submittal_data_cache))
            self.submittal_data_cache.mkdir(parents=True, exist_ok=True)
        
        for eq in scoped_eq:
            for eq_inst in eq.instances:
                if eq_inst.design_uid is None or eq_inst.design_data is None or eq_inst.submittal_data is None:
                    logger.warning(f"Skipping {eq.name} ({eq_inst.name}) because it is missing design uid, design data, or submittal data: {eq_inst.design_uid}, {eq_inst.design_data}, {eq_inst.submittal_data}")
                    continue
                else:
                    logger.info(f"Analyzing {eq.name} ({eq_inst.name}, {eq_inst.design_uid})")
                    # lookup cached first
                    spec_results = self.get_spec_results_for_eq_instance(eq_inst, run_manager=run_manager, **kwargs)
                    # if spec_results is good, then analyze them
                    spec_analysis = self.analyze_spec_results_for_eq_instance(eq_inst, spec_results, run_manager=run_manager, **kwargs)    

        if run_manager:
            run_manager.on_text("Log something about this run")

        return {self.output_key: {}}
    
    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        raise NotImplementedError("Async not implemented")

    @property
    def _chain_type(self) -> str:
        return "AnalyzeSpecsChain"
    
    def get_spec_results_for_eq_instance(self, eq_inst: ScopedEquipmentInstance, spec_defs: Dict[str, str], run_manager: Optional[CallbackManagerForChainRun] = None, **kwargs) -> List[SpecificationResults]:
        """Analyze the specs for a single equipment instance.
        """

        if eq_inst.design_uid is None or eq_inst.design_data is None or eq_inst.submittal_data is None:
            logger.warning(f"Skipping {eq_inst.name} because it is missing design uid, design data, or submittal data: {eq_inst.design_uid}, {eq_inst.design_data}, {eq_inst.submittal_data}")
            return {}

        logger.info(f"Getting spec results for {eq_inst.name} ({eq_inst.design_uid})")

        parser = PydanticOutputParser(pydantic_object=SpecificationResults)
        prompt = PromptTemplate(
            template="Answer the user query.\n{format_instructions}\n{query}\n{data}",
            input_variables=["query", "data"],
            partial_variables={
                "format_instructions": parser.get_format_instructions(),
                "design_data": eq_inst.design_data,
                "submittal_data": eq_inst.submittal_data,
                }
        )
        spec_results: List[SpecificationResults] = []
        for spec_name, spec_def in spec_defs:
            query = f"What is the {spec_name} for {eq_inst.design_uid}? The {spec_name} is {spec_def}."
            logger.debug(f"Querying LLM with query:\n{query}")
            _input = prompt.format_prompt(query=query)

            try:
                output = self.chat.predict(_input.to_string())
                parsed_output = parser.parse(output)
            except Exception as e:
                logger.warning(f"LLM couldn't parse output for {eq_inst.name} ({eq_inst.design_uid}) spec {spec_name}:\n{output}")
                parsed_output = SpecificationResults(
                    eq_uid=eq_inst.design_uid,
                    spec_name=spec_name,
                    design_result=None,
                    submittal_result=None,
                    confidence=None,
                    notes=f"LLM couldn't parse output for {eq_inst.name} ({eq_inst.design_uid}) spec {spec_name}:\n{output}",
                )
                    
            spec_results.append(parsed_output)
        return spec_results
    
    def analyze_spec_results_for_eq_instance(self, eq_inst: ScopedEquipmentInstance, spec_results: List[SpecificationResults], spec_defs: Dict[str, str], run_manager: Optional[CallbackManagerForChainRun] = None, **kwargs) -> List[SpecificationAnalysis]:

        logger.info(f"Getting spec results for {eq_inst.name} ({eq_inst.design_uid})")

        parser = PydanticOutputParser(pydantic_object=SpecificationResults)
        prompt = PromptTemplate(
            template="Answer the user query.\n{format_instructions}\n{query}\n{data}",
            input_variables=["query", "data"],
            partial_variables={
                "format_instructions": parser.get_format_instructions(),
                }
        )
        spec_analysis: List[SpecificationAnalysis] = []
        for spec_res in spec_results:
            query = f"The engineering design document for {eq_inst.design_uid} specify that the {spec_res.spec_name} is {spec_res.design_result}. The contractor submittal document specifies that the {spec_res.spec_name} is {spec_res.submittal_result}. Based on the definition of {spec_res.spec_name}, what should the {spec_res.spec_name} be and should we make any notes on the design or submittal about the results?\n{spec_defs[spec_res.spec_name]}"
            logger.debug(f"Querying LLM with query:\n{query}")
            _input = prompt.format_prompt(query=query)

            try:
                output = self.chat.predict(_input.to_string())
                parsed_output = parser.parse(output)
            except Exception as e:
                logger.warning(f"LLM couldn't analyze spec results for {eq_inst.name} ({eq_inst.design_uid}) spec {spec_res.spec_name}:\n{output}")
                logger.exception(f"LLM error")
                parsed_output = SpecificationAnalysis(
                    eq_uid=eq_inst.design_uid,
                    spec_name=spec_res.spec_name,
                    design_result=spec_res.design_result,
                    submittal_result=spec_res.submittal_result,
                    analysis_result=None,
                )
        
            spec_analysis.append(parsed_output)
        return spec_analysis