from __future__ import annotations
import json
import re
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from pydantic import Extra, root_validator, Field
from loguru import logger

from langchain.schema import BaseMessage
from langchain.chains.base import Chain
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.prompts.chat import ChatPromptTemplate

from meche_copilot.schemas import AgentConfig
from meche_copilot.chains.helpers.specs_retriever import SpecsRetriever
from meche_copilot.utils.chunk_dataframe import chunk_dataframe, combine_dataframe_chunks
from meche_copilot.utils.envars import OPENAI_API_KEY

class LookupSpecsChain(Chain):
    """Lookup spec in design and submittal docs and compare against spec info from template"""

    doc_retriever: AgentConfig
    spec_reader: AgentConfig
    output_key: str = "result" #: :meta private:

    chat: Optional[ChatOpenAI]

    class Config:
        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Will be whatever keys the prompt expects.

        :meta private:
        """
        all_keys = (
            self.doc_retriever.system_prompt_template.input_variables
            + self.doc_retriever.message_prompt_template.input_variables
            + self.spec_reader.system_prompt_template.input_variables
            + self.spec_reader.message_prompt_template.input_variables
        )

        all_keys = list(set(all_keys))

        logger.debug(f"input variables: {all_keys}")
        return all_keys
    
    @property
    def output_keys(self) -> List[str]:
        """Will always return text key.

        :meta private:
        """
        return [self.output_key]

    # TODO - add a build extra validator??

    def _validate_inputs(self, inputs: Dict[str, Any]) -> None:
        """Overide to skip validation"""
        # TODO - skipping this validation cuzza for now (should check to make sure required input keys are present once I've settled on what they are)

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:

        logger.debug(f"input keys: {inputs.keys()}")
        
        retriever = SpecsRetriever(doc_retriever=self.doc_retriever, source=inputs.get('source'))
        relavent_docs = retriever.get_relevant_documents(
            query="", # TODO - this is annoying
            refresh_source_docs=inputs.get("refresh_source_docs", False)
        )
        if len(relavent_docs) <= 0:
            raise ValueError("Doc retreiver couldn't find any relavent docs. Exiting chain.")
        else:
            relavent_ref_docs_as_dicts = [doc.dict() for doc in relavent_docs]
            relavent_ref_docs_as_string = json.dumps(relavent_ref_docs_as_dicts)  # Convert to JSON string
            inputs['relavent_docs'] = relavent_ref_docs_as_string
            logger.info(f"Doc retreiver found {len(relavent_docs)} relavent docs.")


        default_model_name = "gpt-4" # 8192 tokens
        model_name = self.spec_reader.model_name or default_model_name
        self.chat = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model=model_name, callbacks=run_manager.get_child() if run_manager else None)

        # generate prompts chunked according to model token limits
        chat_prompt_chunks = self._generate_chat_prompt_chunks(
            chat=self.chat,
            inputs=inputs
        )

        res_chunks = []
        logger.debug(f"Chunked prompt into {len(chat_prompt_chunks)} chunks to fit tokens limits")
        for i in range(len(chat_prompt_chunks)):
            try:
                messages = chat_prompt_chunks[i]
                logger.debug(f"Sending prompt: {messages}")
                res = self.chat(
                    messages=messages, 
                    callbacks=run_manager.get_child() if run_manager else None
                )
                if run_manager:
                    run_manager.on_text("TODO - Log something about this run")

                logger.debug(f"res={res.content}")
                # TODO - use pydantic or .construct to check json format? or maybe just check shapes?
                json_res = self._extract_and_validate_json(res.content)
                res_chunks.append(json_res)
            except Exception as e:
                logger.error(f"Error in chat: {e}")
                raise e

        final_res = None
        try:
            logger.info(f"Recombining results...")
            final_res = combine_dataframe_chunks(dfs=res_chunks)
        except Exception as e:
            logger.error(f"Couldn't combine df chunks")

        return {self.output_key: final_res}
    
    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        raise NotImplementedError("Async not implemented")

    @property
    def _chain_type(self) -> str:
        return "SpecsLookupChain"
    
    def _generate_chat_prompt_chunks(self, inputs: Dict[str, Any], chat: ChatOpenAI) -> List[List[BaseMessage]]:
        chat_prompt_chunks = []
        # spec_def_df: pd.DataFrame = inputs.get('spec_def_df')
        spec_res_df: pd.DataFrame = inputs.get('spec_res_df')
        relavent_docs = inputs.get('relavent_docs')

        # Make sure spec_def_df row names equal the columns of spec_res_df
        # if list(spec_def_df.index) != list(spec_res_df.columns):
        #     raise ValueError("Number of rows in spec_def_df must be equal to the number of columns in spec_res_df")

        chat_prompt = ChatPromptTemplate.from_messages([
                    self.spec_reader.system_prompt_template,
                    self.spec_reader.message_prompt_template,
                ])
        logger.debug(f"Chat prompt template created with inputs: {chat_prompt.input_variables}")

        start_idx = 0
        max_tokens = 8192 # TODO - figure out how to lookup max tokens per model?
        while start_idx < spec_res_df.shape[1]:
            logger.debug(f"start_idx={start_idx}")
            # iterate over rows/cols until max_tokens is reached, then append the chunk and update start index and do the same for the next chunk
            messages = None
            prev_messages = None
            tokens = None
            prev_tokens = None
            # TODO - update this to keep the first column (ie name always there...)
            for i in range(start_idx, spec_res_df.shape[1]):
                prev_messages = messages # save the prev messages
                messages = chat_prompt.format_prompt(
                    ref_docs=relavent_docs,
                    # spec_defs=spec_def_df.iloc[start_idx:i+1].to_json(),
                    spec_results=spec_res_df.iloc[:, start_idx:i+1].to_json(),
                ).to_messages()
                prev_tokens = tokens # save the prev num tokens
                tokens = chat.get_num_tokens_from_messages(messages)
                logger.info(f"Number of tokens in messages: {tokens}")
                if tokens > max_tokens:
                    if prev_messages is not None:
                        logger.debug(f"Adding chunk with token size: {prev_tokens}")
                        chat_prompt_chunks.append(prev_messages)
                    start_idx = i + 1 # update start idx
                    break
            else: 
                # if loop completes without breaking (ie. all remaining data fits within max_tokens)
                chat_prompt_chunks.append(messages)
                start_idx = spec_res_df.shape[1] # update start_idx to exit the while loop
        
        if not chat_prompt_chunks:
            raise ValueError("The base prompt is too large and exceeds the maximum token limit.")

        return chat_prompt_chunks

    # TODO - shouldn't this be done with an output parser?
    def _extract_and_validate_json(self, input_string):
        # Use a regular expression to find the outermost bracketed JSON object inside the string
        match = re.search(r'{.*}', input_string)
        if match:
            json_string = match.group(0)
            try:
                # Try to load the JSON object to validate it
                json_object = json.loads(json_string)
                logger.debug("Valid JSON")
                return json_object
            except json.JSONDecodeError:
                logger.error("Invalid JSON")
        else:
            logger.error("No JSON found in the input string")
            return {}