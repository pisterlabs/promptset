from __future__ import annotations
import openai
from langchain.agents import Tool
import os

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import asyncio
from typing import Any, Dict, List, Optional
from pydantic import Extra
from langchain.schema import BaseLanguageModel
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from langchain.schema import BasePromptTemplate
import json

async def backend_chain(
        inputs: Dict[str, Any],
        llm: BaseLanguageModel,
        advanced_llm: BaseLanguageModel,
    ) -> str:
        # Feature Extraction
        prompt = PromptTemplate(
            input_variables=["project_details", "project_technologies"],
            template="""
            Given the following project description and tech stack, identify and elaborate on the key backend features that would be necessary for development. The backend features should not involve any frontend components or styling. The backend features should be described in terms of capabilities.
            This project is a hackathon project. Break apart the features into MVP and additional features. The MVP should be the minimum features necessary to have a working prototype.
            Project description: {project_details}
            Technologies: {project_technologies}
            """
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        backend_features = await chain.arun(inputs)
        
        # Specification Creation
        specification_prompt = PromptTemplate(
            input_variables=["backend_features", "project_technologies"],
            template="""
            Given the extracted backend features and the specified skills/technologies, create a detailed technical specification. 
            This specification should include the technologies to be used, the architecture, the different routes/endpoints, their inputs and outputs, and any potential hardware and startup costs.
            However, they should be split into two categories: MVP and additional features. The MVP should be the minimum features necessary to have a working prototype.
            You should ignore the technologies for the frontend and focus on the backend.
            Please also mention any other technical considerations.
            
            Backend Features: {backend_features}
            
            Project Technologies: {project_technologies}
            """
        )
        specification_chain = LLMChain(llm=llm, prompt=specification_prompt)
        specification = await specification_chain.arun({
            'backend_features': backend_features,
            'project_technologies': inputs['project_technologies']
        })
        # Approval check
        approval_prompt = PromptTemplate(
            input_variables=["technical_specification", "aspect", "group_size", "group_experience"],
            template="""
            Given the developed technical specification, conduct a thorough review of the MVP Features only for any inconsistencies or issues. 
            Also, evaluate whether the MVP Features, can be realistically completed within the two day hackathon for {group_size} people, considering the complexity and the technology stack required.
            
            The MVP Features are specifically listed under the heading 'MVP Features'. 
            Please completely disregard any features or sections listed under 'Additional Features' or any similar headers.
            This specification is only for the {aspect} aspect of the project, and should not be evaluated for other aspects.

            Answer this question: Can the MVP Features be realistically completed within the two day hackathon for {group_size} people with this skill level: {group_experience}?
            Output only a json with keys 'approval' and 'comments'. 
            If yes, the value of 'approval' should be '1' and the value of 'comments' should be an empty string
            If not, the value of 'approval' should be '0' and the value of 'comments' should be a string with the issues and inconsistencies listed.

            Technical Specification: {technical_specification}
            
            Output only a json with keys 'approval' and 'comments'. 
            """
        )
        
        approval_chain = LLMChain(llm=advanced_llm, prompt=approval_prompt)
        approval = await approval_chain.arun({
            'technical_specification': specification,
            'aspect': 'backend',
            'group_size': inputs['group_size'],
            'group_experience': inputs['group_experience']
        })
        
        approvals_object = json.loads(approval)
        
        return_obj = {
            'approval': approvals_object['approval'],
            'comments': approvals_object['comments'],
            'idea': inputs['project_details'],
            'features': backend_features,
            'specifications': specification
        }
        
        return json.dumps(return_obj)