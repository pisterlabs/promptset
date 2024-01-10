"""
A chain that reads engineering design documents and extracts data from them by reading schedules and drawings
"""
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

from meche_copilot.schemas import Source, ScopedEquipment, EngineeringDesignSchedule
from meche_copilot.chains.helpers.mechanical_schedule_table_to_df import mechanical_schedule_table_to_df
from meche_copilot.utils.converters import pydantic_from_jsonl, pydantic_to_jsonl, title_to_filename
from meche_copilot.pdf_helpers.get_pages_from_text import get_pages_from_text
from meche_copilot.utils.envars import OPENAI_API_KEY, DATA_CACHE

# TODO - everywhere a llm/prompt is used, it should include a way to chunk the data in case it is too long for the model
# TODO - save llm conversation to cache

class ReadDesignChain(Chain):
    """
    A chain that reads engineering design documents and extracts data from them by reading schedules and drawings

    Step 1: Get all potential schedules from the design documents
    Step 2: Select schedules relavent to the scoped equipment
    Step 3: Extract schedule metadata (row labels, remarks, etc)
    Step 4: Extract schedule rows
    Step 5: Extract schedule column labels
    Step 6: Extract schedule table using camelot using the metadata from the previous steps so camelot can extract even complex tables robustly
    """
    
    prompt: BasePromptTemplate = PromptTemplate.from_template('') # TODO - use build extras?
    chat = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model="gpt-4")
    design_data_cache: Path = DATA_CACHE / 'design_data'
    design_schedules_fpath = design_data_cache / 'design_schedules.jsonl'
    design_drawings_fpath = design_data_cache / 'design_drawings.jsonl'
    output_key: str = "result" #: :meta private:

    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True

    @validator('design_data_cache', 'design_schedules_fpath', 'design_drawings_fpath', pre=True)
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

        Looks for cached design data, and if it doesn't exist, reads design schedules and drawings and creates design data.
        """

        logger.debug(f"input keys: {inputs.keys()}")
        scoped_eq: List[ScopedEquipment] = inputs.get('scoped_eq', [])
        refresh_design_data: bool = inputs.get('refresh_design_data', False)
        kwargs = getattr(self, 'kwargs', {})

        show_your_work: bool = kwargs.get('show_your_work', False)
        if show_your_work:
            logger.add(sink=self.design_data_cache / 'logs/read_design_chain.log', rotation="1 week", level="DEBUG")

        if refresh_design_data: # remove cached
            logger.info(f"Removing cached design data...")
            os.remove(str(self.design_data_cache))
            self.design_data_cache.mkdir(parents=True, exist_ok=True)
            design_df = None

        design_schedules = None
        logger.info(f"Looking up cached design schedules...")
        try:
            if self.design_schedules_fpath.exists():
                logger.debug(f"Using cached design schedules.")
                design_schedules = pydantic_from_jsonl(self.design_schedules_fpath, EngineeringDesignSchedule)
            else:
                logger.debug(f"Cached design data not found. Creating new: {self.design_schedules_fpath}")
                design_schedules = self.read_design_schedules(scoped_eq=scoped_eq, show_your_work=show_your_work)
                logger.debug(f"Writing design data to {self.design_schedules_fpath}")
                pydantic_to_jsonl(design_schedules, self.design_schedules_fpath)
        except Exception as e:
            logger.exception(f"Couldn't read design schedules")

        design_drawings = None
        logger.info(f"Looking up cached design drawings...")
        try:
            if self.design_drawings_fpath.exists():
                logger.debug(f"Using cached design drawings.")
                design_drawings = pydantic_from_jsonl(self.design_drawings_fpath, EngineeringDesignSchedule)
            else:
                logger.debug(f"Cached design data not found. Creating new: {self.design_drawings_fpath}")
                design_drawings = self.read_design_drawings(scoped_eq=scoped_eq, show_your_work=show_your_work)
                logger.debug(f"Writing design data to {self.design_drawings_fpath}")
                pydantic_to_jsonl(design_drawings, self.design_drawings_fpath)
        except Exception as e:
            logger.exception(f"Couldn't read design drawings")

        if run_manager:
            run_manager.on_text("Log something about this run")

        return {self.output_key: {"design_schedules": design_schedules, "design_drawings": design_drawings}}
    
    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        raise NotImplementedError("Async not implemented")

    @property
    def _chain_type(self) -> str:
        return "ReadDesignChain"
    
    def read_design_schedules(self, scoped_eq: List[ScopedEquipment], **kwargs):
        """Reads the design schedules from the design documents and extracts the data from them and writes to design_schedules.parquet"""

        # TODO - turn these into steps with try agains and configurable logic about when to go to next step

        show_your_work: bool = kwargs.get('show_your_work', False)

        ### FIND ALL POTENTIAL SCHEDULES IN DESIGN DOCS ###
        # NOTE: necessary to reduce the number of tokens sent to llms later
        all_design_schedules: List[EngineeringDesignSchedule] = []
        all_design_schedules_titles_fpath = self.design_data_cache / '1_all_design_schedules.jsonl'
        logger.info(f"Getting all schedules from design docs...")
        try:
            if all_design_schedules_titles_fpath.exists():
                logger.debug(f"Using cached file: {all_design_schedules_titles_fpath}")
                all_design_schedules = pydantic_from_jsonl(all_design_schedules_titles_fpath, EngineeringDesignSchedule)
            else:
                logger.debug(f"Cached file not found, creating new: {all_design_schedules_titles_fpath}")
                all_design_schedules = self._get_design_schedules(scoped_eq=scoped_eq)
                if len(all_design_schedules) == 0:
                    raise Exception("No schedules found in design documents. Exiting chain.")
                logger.debug(f"Writing to cache: {all_design_schedules_titles_fpath}")
                pydantic_to_jsonl(all_design_schedules, all_design_schedules_titles_fpath)
            logger.debug(f"{len(all_design_schedules)} schedule candidates in design docs")
        except Exception as e:
            logger.exception(f"Couldn't get list of all schedules. Exiting chain.")
            raise e
        logger.success("Done getting all schedules from design docs.")

        scoped_design_schedules: List[EngineeringDesignSchedule] = []

        ### LLM SELECT RELEVANT SCHEDULE TITLES ###
        scoped_design_schedules_fpath = self.design_data_cache / '2_scoped_design_schedules_with_titles.jsonl'
        logger.info(f"Selecting schedules relavent to scoped equipment...")
        try:
            if scoped_design_schedules_fpath.exists():
                logger.debug(f"Using cached file: {scoped_design_schedules_fpath}")
                scoped_design_schedules = pydantic_from_jsonl(scoped_design_schedules_fpath, EngineeringDesignSchedule)
            else:
                logger.info(f"Cached file not found, creating new: {scoped_design_schedules_fpath}")
                scoped_design_schedules = self._select_schedules_from_equipment(scoped_eq=scoped_eq, all_design_schedules=all_design_schedules)
                if len(scoped_design_schedules) == 0:
                    raise Exception("No schedules found in design documents. Exiting chain.")
                logger.info(f"Writting to cache: {scoped_design_schedules_fpath}")
                pydantic_to_jsonl(scoped_design_schedules, scoped_design_schedules_fpath)
        except Exception as e:
            logger.exception(f"Couldn't select relavent schedule titles. Exiting chain.")
            raise e
        logger.success("Done selecting schedules relavent to scoped equipment.")

        ### LLM EXTRACT SCHEDULE METADATA ###
        schedules_with_metadata_fpath = self.design_data_cache / '3_scoped_design_schedules_with_metadata.jsonl'
        logger.info(f"Getting schedule metadata for each schedule")
        try:
            if schedules_with_metadata_fpath.exists():
                logger.debug(f"Using cached file: {schedules_with_metadata_fpath}")
                scoped_design_schedules = pydantic_from_jsonl(schedules_with_metadata_fpath, EngineeringDesignSchedule)
            else:
                logger.info(f"Cached file not found, creating new: {schedules_with_metadata_fpath}")
                scoped_design_schedules = self._extract_schedule_metadata(scoped_design_schedules=scoped_design_schedules)
                if len(scoped_design_schedules) == 0:
                    raise Exception("No schedules found in design documents. Exiting chain.")
                logger.info(f"Writting to cache: {schedules_with_metadata_fpath}")
                pydantic_to_jsonl(scoped_design_schedules, schedules_with_metadata_fpath)
            logger.debug(f"Got metadata for {len(scoped_design_schedules)} schedules")
            for sched in scoped_design_schedules:
                logger.debug(f"Found {len(sched.row_data.keys())} rows for {sched.title} and {'no' if len(sched.remarks) == 0 else 'some'} remarks")
        except Exception as e:
            logger.exception(f"Couldn't extract schedule metadata. Exiting chain.")
            raise e
        logger.success("Done getting schedule metadata.")

        ### LLM EXTRACT SCHEDULE ROWS ###
        schedules_with_row_data_fpath = self.design_data_cache / '4_scoped_design_schedules_with_row_data.jsonl'
        logger.info(f"Getting schedule rows for each schedule")
        try:
            if schedules_with_row_data_fpath.exists():
                logger.debug(f"Using cached file: {schedules_with_row_data_fpath}")
                scoped_design_schedules = pydantic_from_jsonl(schedules_with_row_data_fpath, EngineeringDesignSchedule)
            else:
                logger.info(f"Cached file not found, creating new: {schedules_with_row_data_fpath}")
                scoped_design_schedules = self._extract_schedule_rows(scoped_design_schedules=scoped_design_schedules)
                if len(scoped_design_schedules) == 0:
                    raise Exception("No schedules found in design documents. Exiting chain.")
                logger.debug(f"Writting to cache: {schedules_with_row_data_fpath}")
                pydantic_to_jsonl(scoped_design_schedules, schedules_with_row_data_fpath)
        except Exception as e:
            logger.exception(f"Couldn't extract schedule rows. Exiting chain.")
            raise e
        logger.success("Done getting schedule rows.")

        ### LLM EXTRACT SCHEDULE COLUMN LABELS ###
        schedules_with_column_labels_fpath = self.design_data_cache / '5_scoped_design_schedules_with_column_labels.jsonl'
        logger.info(f"Getting schedule column labels for each schedule")
        try:
            if schedules_with_column_labels_fpath.exists():
                logger.debug(f"Using cached file: {schedules_with_column_labels_fpath}")
                scoped_design_schedules = pydantic_from_jsonl(schedules_with_column_labels_fpath, EngineeringDesignSchedule)
            else:
                logger.info(f"Cached file not found, creating new: {schedules_with_column_labels_fpath}")
                scoped_design_schedules = self._extract_schedule_column_labels(scoped_design_schedules=scoped_design_schedules)
                if len(scoped_design_schedules) == 0:
                    raise Exception("No schedules found in design documents. Exiting chain.")
                logger.debug(f"Writting to cache: {schedules_with_column_labels_fpath}")
                pydantic_to_jsonl(scoped_design_schedules, schedules_with_column_labels_fpath)
        except Exception as e:
            logger.exception(f"Couldn't extract schedule column labels. Exiting chain.")
            raise e
        logger.success("Done getting schedule column labels.")

        ### EXTRACT SCHEDULE TABLE USING CAMELOT ###
        schedules_dfs: List[pd.DataFrame] = []
        logger.info(f"Extracting schedule tables using camelot")
        try:
            cached_fnames = [str(f.name) for f in self.design_data_cache.iterdir() if f.suffix == '.parquet']
            for eds in scoped_design_schedules:
                fname = title_to_filename(eds.title) + '.parquet'
                if fname in cached_fnames:
                    logger.debug(f"Using cached schedule data for {eds.title}: {fname}")
                    schedule_df = pd.read_parquet(self.design_data_cache / fname)
                    schedules_dfs.append(schedule_df)
                else:
                    try:
                        logger.info(f"Extracting schedule data for {eds.title}...")
                        schedule_df = self._extract_schedule_table(eds=eds, show_your_work=show_your_work)
                        schedule_df.to_parquet(self.design_data_cache / fname)
                        schedules_dfs.append(schedule_df)
                    except Exception as e:
                        logger.error(f"Camelot couldn't extract schedule data for {eds.title}. Continuing chain.")
        except Exception as e:
            logger.exception(f"Couldn't extract schedule data. Exiting chain.")
            raise e
        logger.success("Done getting schedule data.")

        ### COMBINE EQUIPMENT SCHEDULE DATA COMPONENTS ###
        logger.info(f"Combining schedule data for each schedule")
        try:
            if self.design_schedules_fpath.exists():
                logger.debug(f"Using cached file: {self.design_schedules_fpath}")
                scoped_design_schedules = pydantic_from_jsonl(self.design_schedules_fpath, EngineeringDesignSchedule)
            else:
                logger.info(f"Cached file not found, creating new: {self.design_schedules_fpath}")
                scoped_design_schedules = self._combine_schedule_results(scoped_design_schedules=scoped_design_schedules)
                if len(scoped_design_schedules) == 0:
                    raise Exception("Combining schedule data failed. Exiting chain.")
                logger.debug(f"Writting to cache: {self.design_schedules_fpath}")
                pydantic_to_jsonl(scoped_design_schedules, self.design_schedules_fpath)
        except Exception as e:
            logger.exception(f"Couldn't combine schedule data. Exiting chain.")
            raise e

    def read_design_drawings(self, scoped_eq: List[ScopedEquipment], **kwargs):
        raise NotImplementedError("read_design_drawings not implemented")
    
    def _get_design_schedules(self, scoped_eq: List[ScopedEquipment], **kwargs) -> List[EngineeringDesignSchedule]:
        """Returns a list of all potential schedules from the design_fpaths"""
        
        design_schedules: List[EngineeringDesignSchedule] = []
        design_fpaths_completed = set()
        for eq in scoped_eq:
            for fpath in eq.design_source.ref_docs:
                cache_fpath = self.design_data_cache / f"{fpath.stem}_text_blocks"
                cache_fpath.mkdir(parents=True, exist_ok=True)
                if fpath in design_fpaths_completed:
                    continue
                else:
                    logger.debug(f"Processing design reference doc: {fpath}")
                    with fitz.open(str(fpath)) as doc:
                        for p in doc:
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

                            for text in page_text_blocks:
                                if "SCHEDULE" in text and len(text.split(' ')) < 10:
                                    design_schedules.append(EngineeringDesignSchedule(
                                        title=text.replace('\n', ' '),
                                        page_number=int(p.number),
                                        fpath=str(fpath),
                                    ))
                    design_fpaths_completed.add(fpath)
        return design_schedules
    
    def _select_schedules_from_equipment(self, scoped_eq: List[ScopedEquipment], all_design_schedules: List[EngineeringDesignSchedule], **kwargs) -> List[EngineeringDesignSchedule]:
        """Returns a list of EquipmentScheduleTitles objects for each equipment in scoped_eq"""

        if len(all_design_schedules) == 0:
            raise Exception("No schedules found in design. Can't select relevant schedules from none")
        
        logger.info(f"LLM is selecting schedules relevant to the equipment...")
        class EquipmentScheduleTitles(BaseModel):
            equipment_name: str = Field(description="Equipment name/type (eg. hydronic pump, exhaust fan, energy recovery ventilator")
            schedule_titles: List[str] = Field(description="Schedule table titles that match the equipment name. There may be 0 or more items in the list")
        
        all_design_schedules_titles = [s.title for s in all_design_schedules]
        parser = PydanticOutputParser(pydantic_object=EquipmentScheduleTitles)
        prompt = PromptTemplate(
            template="Answer the user query.\n{format_instructions}\n{query}\n{data}",
            input_variables=["query"],
            partial_variables={
                "format_instructions": parser.get_format_instructions(),
                "data": '\n\n'.join(all_design_schedules_titles)
                }
        )

        scoped_design_schedules: List[EquipmentScheduleTitles] = []
        for eq in scoped_eq:
            query = f"Match the relavent schedule(s) for the equipment type: {eq.name}, notes: {eq.design_source.notes}"
            logger.info(f"Querying LLM with query: {query}")
            # NOTE: it may be useful to include examples of alt schedules, continuations, what are not examples of schedules, other names for equipment, etc
            _input = prompt.format_prompt(query=query)

            try:
                output = self.chat.predict(_input.to_string())
                parsed_output = parser.parse(output)
                logger.debug(f"LLM found the following relavent schedules for '{eq.name}': {parsed_output}")
            except Exception as e:
                logger.warning(f"LLM couldn't find relavent schedules for '{eq.name}'. Continuing to next equipment.")
                logger.exception(f"LLM error")
                parsed_output = EquipmentScheduleTitles(equipment_name=eq.name, schedule_titles=[])
            
            for schedule_title in parsed_output.schedule_titles:
                eds = [s for s in all_design_schedules if s.title.strip().upper() == schedule_title.strip().upper()] # TODO - this is shitty...be less stupid
                if len(eds) != 1:
                    raise Exception(f"Found {len(eds)} schedules with title: {schedule_title} in all design schedules. Expected 1.")
                scoped_design_schedules.append(EngineeringDesignSchedule(
                    equipment_name=eq.name,
                    title=schedule_title,
                    page_number=eds[0].page_number,
                    fpath=eds[0].fpath,
                ))
        return scoped_design_schedules

    def _extract_schedule_metadata(self, scoped_design_schedules: List[EngineeringDesignSchedule], **kwargs) -> List[EngineeringDesignSchedule]:

        if len(scoped_design_schedules) == 0:
            raise Exception("No equipment schedule titles found. Can't extract schedule metadata.")
        
        class ScheduleMetadata(BaseModel):
            title: str = Field(description="equipment schedule table title (eg. EXAMPLE EQUIPMENT SCHEDULE, EXAMPLE EQUIPMENT SCHEDULE (CONT....), EXAMPLE EQUIPMENT SCHEDULE (ALTERNATE), etc")
            row_labels: List[str] = Field(description="schedule table row labels usually called SYMBOL or MARK (eg. VRF-1, SL-4)")
            remarks: str = Field(description="remarks or notes for the schedule table")
        parser = PydanticOutputParser(pydantic_object=ScheduleMetadata)
        prompt = PromptTemplate(
            template="Answer the user query.\n{format_instructions}\n{query}\n{data}",
            input_variables=["query", "data"],
            partial_variables={
                "format_instructions": parser.get_format_instructions()
                }
        )

        scoped_design_schedules_with_metadata: List[EngineeringDesignSchedule] = []
        for eds in scoped_design_schedules:
            fpath = self.design_data_cache / f"{eds.fpath.stem}_text_blocks" / f"{eds.page_number}.jsonl"
            if not fpath.exists():
                raise Exception(f"Couldn't find text blocks in cache for fpath: {fpath}")
            page_text_blocks = [json.loads(line)["text_block"] for line in fpath.open()]
            query = f"What are the row labels, and remarks for the {eds.title} table?"
            logger.debug(f"Querying LLM with query:\n{query}")
            _input = prompt.format_prompt(query=query, data='\n\n'.join(page_text_blocks))

            try:
                output = self.chat.predict(_input.to_string())
                parsed_output = parser.parse(output)
            except Exception as e:
                logger.warning(f"LLM couldn't find schedule metadata for equipment: {eds.equipment_name}, schedule: {eds.title}. Continuing to next schedule.")
                logger.exception(f"LLM error")
                parsed_output = ScheduleMetadata(schedule_title=eds.equipment_name, row_labels=[], remarks=[])
                    
            row_data = {}
            for row_label in parsed_output.row_labels:
                row_data[row_label] = []
            scoped_design_schedules_with_metadata.append(EngineeringDesignSchedule(
                equipment_name=eds.equipment_name,
                title=eds.title,
                page_number=eds.page_number,
                fpath=eds.fpath,
                remarks=parsed_output.remarks,
                row_data=row_data,
            ))
        return scoped_design_schedules_with_metadata
    
    def _extract_schedule_rows(self, scoped_design_schedules: List[EngineeringDesignSchedule], **kwargs) -> List[EngineeringDesignSchedule]:
        if len(scoped_design_schedules) == 0:
            raise Exception("No equipment schedule metadata found. Can't extract row data without metadata.")
        
        class ScheduleRowData(BaseModel):
            row_data: Dict[str, List[str]] = Field(description="each row label is a key in the dict and the value is a list of the data for that row")
            
            @validator('row_data')
            def same_list_lengths(cls, v):
                """All row values lists should be the same length"""
                row_lengths = [len(row_values) for row_values in v.values()]
                if len(set(row_lengths)) > 1:
                    raise ValueError(f"Row values lists are not the same length: {row_lengths}")
                return v

        parser = PydanticOutputParser(pydantic_object=ScheduleRowData)
        prompt = PromptTemplate(
            template="Answer the user query.\n{format_instructions}\n{query}\n{data}",
            input_variables=["query", "data"],
            partial_variables={
                "format_instructions": parser.get_format_instructions()
                }
        )

        scoped_design_schedules_with_row_data: List[EngineeringDesignSchedule] = []
        for eds in scoped_design_schedules:
            if eds.fpath is None or eds.page_number is None:
                raise Exception(f"This schedule metadata doesn't have a required attributes: (fpath, page_number): {eds}")
            fpath = self.design_data_cache / f"{eds.fpath.stem}_text_blocks" / f"{eds.page_number}.jsonl"
            if not fpath.exists():
                raise Exception(f"Couldn't find text blocks in cache for fpath: {fpath}")
            page_text_blocks = [json.loads(line)["text_block"] for line in fpath.open()]
            query = f"What are the values for each of the row labels in the '{eds.title}' table: {list(eds.row_data.keys())}?"
            logger.debug(f"Querying LLM with query:\n{query}")
            _input = prompt.format_prompt(
                query=query, 
                data='\n\n'.join(page_text_blocks)
                )

            try:
                output = self.chat.predict(_input.to_string())
                parsed_output = parser.parse(output)

                # validate that the number of row labels returned is the same as the number of row labels input
                if len(parsed_output.row_data.keys()) != len(eds.row_data.keys()):
                    raise Exception(f"LLM returned a different number of row labels than expected. Expected: {len(eds.row_data.keys())}, got: {len(parsed_output.row_data.keys())}")

            except Exception as e:
                logger.warning(f"LLM couldn't find schedule row data for schedule: {eds.title}. Continuing to next schedule.")
                logger.exception(f"LLM error")
                parsed_output = None

            scoped_design_schedules_with_row_data.append(EngineeringDesignSchedule(
                equipment_name=eds.equipment_name,
                title=eds.title,
                page_number=eds.page_number,
                fpath=eds.fpath,
                remarks=eds.remarks,
                row_data=parsed_output.row_data if parsed_output else eds.row_data,
            ))

        return scoped_design_schedules_with_row_data
    
    def _extract_schedule_column_labels(self, scoped_design_schedules: List[EngineeringDesignSchedule], **kwargs) -> List[EngineeringDesignSchedule]:
        if len(scoped_design_schedules) == 0:
            raise Exception("No equipment schedule metadata found. Can't extract column labels data without metadata.")
        
        class ScheduleColumnLabels(BaseModel):
            column_labels: List[str] = Field(description="schedule table column headers")

        parser = PydanticOutputParser(pydantic_object=ScheduleColumnLabels)
        prompt = PromptTemplate(
            template="Answer the user query.\n{format_instructions}\n{query}\n{data}",
            input_variables=["query", "data"],
            partial_variables={
                "format_instructions": parser.get_format_instructions()
                }
        )

        scoped_design_schedules_with_column_labels: List[EngineeringDesignSchedule] = []
        for eds in scoped_design_schedules:
            if eds.fpath is None or eds.page_number is None:
                raise Exception(f"This schedule metadata doesn't have a required attributes: (fpath, page_number): {eds}")
            fpath = self.design_data_cache / f"{eds.fpath.stem}_text_blocks" / f"{eds.page_number}.jsonl"
            if not fpath.exists():
                raise Exception(f"Couldn't find text blocks in cache for fpath: {fpath}")
            page_text_blocks = [json.loads(line)["text_block"] for line in fpath.open()]
            num_cols = len(next(iter(eds.row_data.values())))
            query = f"What are the {num_cols} column labels for the '{eds.title}' table?"
            logger.debug(f"Querying LLM with query:\n{query}")
            _input = prompt.format_prompt(
                query=query, 
                data='\n\n'.join(page_text_blocks)
                )

            try:
                output = self.chat.predict(_input.to_string())
                parsed_output = parser.parse(output)
            except Exception as e:
                logger.warning(f"LLM couldn't find schedule column label data for schedule: {eds.title}. Continuing to next schedule.")
                logger.exception(f"LLM error")
                parsed_output = ScheduleColumnLabels(column_labels=[])

            scoped_design_schedules_with_column_labels.append(EngineeringDesignSchedule(
                equipment_name=eds.equipment_name,
                title=eds.title,
                page_number=eds.page_number,
                fpath=eds.fpath,
                remarks=eds.remarks,
                row_data=eds.row_data,
                column_labels=parsed_output.column_labels,
            ))

        return scoped_design_schedules_with_column_labels

    def _extract_schedule_table(self, eds: EngineeringDesignSchedule, **kwargs) -> pd.DataFrame:
        """Extracts the schedule data from the schedule metadata"""

        show_your_work = kwargs.get('show_your_work', False)
        
        if len(eds.row_data.keys()) == 0:
            raise Exception(f"No row labels found for schedule: {eds.title}. Can't extract schedule data.")
        
        if eds.fpath is None or eds.page_number is None or eds.title is None:
            raise Exception(f"This schedule metadata doesn't have a required attributes: (fpath, page_number): {eds}")

        logger.debug(f"Getting mechanical schedule table data for: {eds.title} (p.{eds.page_number})")
        df = mechanical_schedule_table_to_df(
            pdf_fpath=eds.fpath, 
            title=eds.title, 
            last_row=list(eds.row_data.keys())[-1],
            page_number=eds.page_number,
            show_your_work=show_your_work
        )


        logger.debug(f"Postprocessing camelot schedule table data for: {eds.title}")
        cleaned_df = df.iloc[1:, 1:].copy() # remove first col and first row

        # 
        # get indexes of rows that match row data list
        
        # drop all rows below them
        # concat all rows above them 
        logger.debug(f"Done postprocessing camelot schedule table data for: {eds.title}")

        return df

    def _combine_schedule_results(self, scoped_design_schedules: List[EngineeringDesignSchedule], **kwargs) -> List[EngineeringDesignSchedule]:
        """Tries to make best decision possible about how to combine llm and algo results given data obtained in prev steps"""

        # NOTE: so far empirical testing shows that if LLM and camelot disagree about column headers and row data that LLM is better at getting row data and camelot is better at getting column headers

        if len(scoped_design_schedules) == 0:
            raise Exception(f"Can't combine schedule data without no input schedules: {len(scoped_design_schedules)}")

        final_design_schedules: List[EngineeringDesignSchedule] = []
        for eds in scoped_design_schedules:
            logger.debug(f"Combining schedule llm and algo results for: {eds.title} (p.{eds.page_number})")

            res_fpath = self.design_data_cache / f"{title_to_filename(eds.title)}_results.csv"
            if res_fpath.exists():
                res_df = pd.read_csv(res_fpath, index_col=0)
                logger.debug(f"Found results data in cache for: {res_fpath}")
            else:
                logger.debug(f"Cached results data not found. Generating new: {res_fpath}")
                res_df = pd.DataFrame(index=["num_rows", "num_cols", "row_labels", "col_labels"], columns=["llm", "camelot"])
                res_df.fillna(0, inplace=True)

                # get llm rows and cols, if available
                if eds.row_data is not None:
                    keys_list = list(eds.row_data.keys())
                    res_df.loc["num_rows", "llm"] = len(keys_list)
                    if len(keys_list) > 0:
                        res_df.loc["num_cols", "llm"] = len(eds.row_data[keys_list[0]])
                    res_df.loc["row_labels", "llm"] = str(keys_list)
                if eds.headers is not None:
                    res_df.loc["col_labels", "llm"] = str(eds.headers)
                
                # get camelot rows and cols, if available
                fpath = self.design_data_cache / f"{title_to_filename(eds.title)}.parquet"
                if fpath.exists():
                    df = pd.read_parquet(fpath)
                    res_df.loc["num_rows", "camelot"] = df.shape[0]
                    res_df.loc["num_cols", "camelot"] = df.shape[1]
                    res_df.loc["row_labels", "camelot"] = str(df.index.tolist())
                    res_df.loc["col_labels", "camelot"] = str(df.columns.tolist())
                
                # cache results decision matrix
                res_fpath = self.design_data_cache / f"{title_to_filename(eds.title)}_results.csv"

                logger.debug(f"Writing to cache: {res_fpath}")
                res_df.to_csv(res_fpath)

            # use res_df to decide which data to use
            row_data = None
            column_labels = None

            # TODO - decide which data to use

            final_design_schedules.append(EngineeringDesignSchedule(
                equipment_name=eds.equipment_name,
                title=eds.title,
                page_number=eds.page_number,
                fpath=eds.fpath,
                remarks=eds.remarks,
                row_data=row_data,
                column_labels=column_labels,
            ))

        return final_design_schedules