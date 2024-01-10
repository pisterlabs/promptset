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

from meche_copilot.schemas import ScopedEquipment, SubmittalData, EngineeringDesignSchedule
from meche_copilot.utils.converters import pydantic_from_jsonl, pydantic_to_jsonl, title_to_filename
from meche_copilot.utils.envars import OPENAI_API_KEY, DATA_CACHE

# TODO - everywhere a llm/prompt is used, it should include a way to chunk the data in case it is too long for the model

class ReadSubmittalChain(Chain):
    """A chain that reads engineering design documents and extracts data from them by reading schedules and drawings"""
    
    prompt: BasePromptTemplate = PromptTemplate.from_template('') # TODO - use build extras?
    chat = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model="gpt-4")
    submittal_data_cache: Path = DATA_CACHE / 'submittal_data'
    submittal_datas_fpath: Path = submittal_data_cache / 'submittal_datas.jsonl'
    output_key: str = "result" #: :meta private:

    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True

    @validator('submittal_data_cache', pre=True)
    def fpaths_exist(cls, v):
        v.mkdir(parents=True, exist_ok=True)
        return v

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
        
        submittal_datas: List[SubmittalData] = []

        logger.info(f"Reading submittal data")
        try: 
            if self.submittal_datas_fpath.exists():
                logger.debug(f"Using cached submittal data")
                submittal_datas = pydantic_from_jsonl(self.submittal_datas_fpath, SubmittalData)
            else:
                logger.debug(f"Cached submittal data not found. Creating new: {self.submittal_datas_fpath}")
                submittal_datas = self.read_submittal_data(scoped_eq=scoped_eq, **kwargs)
                logger.debug(f"Writing submittal data to cache: {self.submittal_datas_fpath}")
                pydantic_to_jsonl(submittal_datas, self.submittal_datas_fpath)
        except Exception as e:
            logger.exception(f"Couldn't read submittal data.")

        if run_manager:
            run_manager.on_text("Log something about this run")

        return {self.output_key: {"submittal_datas": submittal_datas}}
    
    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        raise NotImplementedError("Async not implemented")

    @property
    def _chain_type(self) -> str:
        return "ReadSubmittalChain"
    
    def read_submittal_data(self, scoped_eq: List[ScopedEquipment], **kwargs):
        """Get submittal data by EDS mark"""

        show_your_work: bool = kwargs.get('show_your_work', False)

        submittal_data: List[SubmittalData] = []
        submittal_fpaths_completed = set()
        for eq in scoped_eq:

            # get the submittal reference docs for each piece of equipment
            for fpath in eq.submittal_source.ref_docs:
                cache_fpath = self.submittal_data_cache / f"{fpath.stem}_text_blocks"
                cache_fpath.mkdir(parents=True, exist_ok=True)
                if fpath in submittal_fpaths_completed:
                    continue
                else:
                    logger.debug(f"Processing submittal reference doc: {fpath}")
                    with fitz.open(str(fpath)) as doc:
                        for p in doc: # load or process each page
                            page_fpath = cache_fpath / f"{p.number}.jsonl"
                            if page_fpath.exists():
                                page_text_blocks = [json.loads(line)["text_block"] for line in page_fpath.open()]
                            else:
                                blocks = p.get_text_blocks()
                                page_text_blocks = []
                                with page_fpath.open('w') as f:
                                    for b in blocks:
                                        page_text_blocks.append(b[4])
                                        f.write(json.dumps({"text_block": str(b[4])}) + '\n')


                            # given page_text_blocks for a eq ref doc
                            for eq_instance in eq.instances:
                                if eq_instance.design_uid is not None:
                                    for text_block in page_text_blocks:
                                        if eq_instance.design_uid in text_block: # if design uid is mentioned anywhere on the page
                                            logger.debug(f"Found submittal data for {eq.name} instance {eq_instance.design_uid} on page {p.number}: {text_block}")
                                            data = {}
                                            data[str(p.number)] = page_text_blocks
                                            submittal_data.append(SubmittalData(
                                                equipment_name=eq.name,
                                                equipment_uid=eq_instance.design_uid,
                                                page_number=str(p.number),
                                                fpath=str(fpath),
                                                data=data
                                            ))
                                            break # break out of text block loop
                    submittal_fpaths_completed.add(fpath)
        return submittal_data