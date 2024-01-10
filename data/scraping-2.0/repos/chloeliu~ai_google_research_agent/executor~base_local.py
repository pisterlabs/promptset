from abc import abstractmethod
from typing import Any
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from langchain.callbacks.manager import Callbacks
from langchain.chains.base import Chain

from schema_local import StepResponse
from langchain.pydantic_v1 import BaseModel, validator
from langchain.schema import (
    AgentAction,
    AgentFinish,
    BaseOutputParser,
    BasePromptTemplate,
    OutputParserException,
)
from typing import List, Callable
# from utils import print_status#status , step_list
import streamlit as st



class BaseExecutor(BaseModel):
    """Base executor."""

    @abstractmethod
    def step(
        self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any
    ) -> StepResponse:
        """Take step."""

    @abstractmethod
    async def astep(
        self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any
    ) -> StepResponse:
        """Take async step."""


class ChainExecutor(BaseExecutor):
    """Chain executor."""
    chain: Chain
    # callbacks: List[Callable] = []
    
    """The chain to use."""
    return_intermediate_steps: bool = False
    @validator('chain', pre=True, always=True)
    def validate_chain(cls, value):
        # Perform your validation logic here. 
        # For example, check if value is an instance of Chain.
        assert isinstance(value, Chain), 'chain must be an instance of Chain'
        return value

    def step(
        self, inputs: dict,  callbacks: Callbacks = None,  **kwargs: Any 
    ) -> StepResponse:
        """Take step."""
        print("here in step")
        print(inputs)
        print('-'*50)
        response = self.chain.__call__(inputs, callbacks=callbacks)
        if 'agent_scratchpad' not in inputs:
            inputs['agent_scratchpad'] = ''
        status=st.status(inputs['current_step'].value, expanded=True,state='running')
        # print("____HERE!! for memory______")
        # print(self.chain.memory)
        # print("____________________________")
        # print("____HERE!! for prompts inputs______")
        # print(inputs)
        # prompts, _ = self.chain.agent.llm_chain.prep_prompts([inputs])
        # print(str(prompts[0]).replace('\\n', '\n'))
        # print("____________________________")
        # print("HERE for response")
        # print(response)
        # print("____________________________")       
        text_outputs = []
        for step in response['intermediate_steps']:
         if step is None:
           text_outputs.append(None)
         elif isinstance(step[0], AgentAction):
            action = step[0].tool
            params = step[0].tool_input  
            ###actions to be added to the status bar
            if action == 'search':
                sentence = f"Googled: " + params['query'].replace('site\n','')
            elif action == 'scrape_website':  
                sentence = f"Read: " +params['url'].replace('\n','')
            status.write("\u2192     "+sentence)    
            text_outputs.append(step[1])   
        status.update(expanded=False,state="complete")
        return StepResponse(response=response['output'], intermediate_steps=text_outputs)
        # return (StepResponse(response=response),self.chain.agent.memory)

    async def astep(
        self, inputs: dict, callbacks: Callbacks = None, **kwargs: Any
    ) -> StepResponse:
        """Take step."""
        response = await self.chain.arun(**inputs, callbacks=callbacks)
        return (StepResponse(response=response),self.chain.agent.memory)