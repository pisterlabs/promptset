#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 C5ailabs Team (Authors: Rohit Sroch) All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Intractive AI Examiner for LEAP platform
"""
#lang chain
import langchain
from langchain import PromptTemplate, LLMChain
from langchain.cache import InMemoryCache

from core.prompt import (
    EXAMINER_ASK_QUESTION_PROMPT,
    EXAMINER_EVALUATE_STUDENT_ANSWER_PROMPT,
    EXAMINER_HINT_MOTIVATE_STUDENT_PROMPT
)

from abc import ABC, abstractmethod
from pydantic import  Extra, BaseModel
from typing import List, Optional, Dict, Any

from utils.logging_handler import Logger

#langchain.llm_cache = InMemoryCache()

class BaseAIExaminer(BaseModel, ABC):
    """Base AI Examiner interface"""

    @abstractmethod
    async def examiner_ask_question(
        self,
        context: str,
        question_type: str,
        topic: str
    ) -> Dict:
        """AI Examiner generates a question"""

    @abstractmethod
    async def examiner_eval_answer(
        self,
        ai_question: str,
        student_solution: str,
        topic: str
    ) -> Dict:
        """AI Examiner evaluates student solution"""
    
    @abstractmethod
    async def examiner_hint_motivate(
        self, 
        ai_question: str,
        student_solution: str,
        topic: str
    ) -> Dict:
        """AI Examiner hints and motivate student"""

class InteractiveAIExaminer(BaseAIExaminer):
    """Interactive AI Examiner"""

    llm_chain_ask_ques: LLMChain
    prompt_template_ask_ques: PromptTemplate

    llm_chain_eval_ans: LLMChain
    prompt_template_eval_ans: PromptTemplate

    llm_chain_hint_motivate: LLMChain
    prompt_template_hint_motivate: PromptTemplate

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
    
    @classmethod
    def load(cls, 
            llm: Any,
            **kwargs):
        """
          Args:
            llm (str): Large language model object
        """

        verbose = kwargs.get("verbose", True)
        # prompt templates
        prompt_template_ask_ques = PromptTemplate(
            template=EXAMINER_ASK_QUESTION_PROMPT, 
            input_variables=[
                "context", 
                "question_type", 
                "topic"
            ]
        )
        prompt_template_eval_ans = PromptTemplate(
            template=EXAMINER_EVALUATE_STUDENT_ANSWER_PROMPT, 
            input_variables=[
                "ai_question", 
                "student_solution", 
                "topic"
            ]
        )
        prompt_template_hint_motivate = PromptTemplate(
            template=EXAMINER_HINT_MOTIVATE_STUDENT_PROMPT, 
            input_variables=[
                "ai_question", 
                "student_solution", 
                "topic"
            ]
        )
        # llm chains
        llm_chain_ask_ques = LLMChain(
            llm=llm, 
            prompt=prompt_template_ask_ques, 
            verbose=verbose
        )
        llm_chain_eval_ans = LLMChain(
            llm=llm, 
            prompt=prompt_template_eval_ans, 
            verbose=verbose
        )
        llm_chain_hint_motivate = LLMChain(
            llm=llm, 
            prompt=prompt_template_hint_motivate, 
            verbose=verbose
        )

        return cls(
            llm_chain_ask_ques=llm_chain_ask_ques, 
            prompt_template_ask_ques=prompt_template_ask_ques, 
            llm_chain_eval_ans=llm_chain_eval_ans,
            prompt_template_eval_ans=prompt_template_eval_ans, 
            llm_chain_hint_motivate=llm_chain_hint_motivate,
            prompt_template_hint_motivate=prompt_template_hint_motivate
        )
    
    async def examiner_ask_question(
        self,
        context: str,
        question_type: str,
        topic: str
    ) -> Dict:
        """AI Examiner generates a question"""
        is_predicted = True
        result = {
            "prediction": None,
            "error_message": None
        }
        try:
            output = self.llm_chain_ask_ques.predict(
                context=context, 
                question_type=question_type,
                topic=topic)
            output = output.strip()
            result["prediction"] = {
                "ai_question": output
            }
            
            return (is_predicted, result)
        except Exception as err:
            Logger.error("Error: {}".format(str(err)))
            is_predicted = False
            result["error_message"] = str(err)
            return (is_predicted, result)

    async def examiner_eval_answer(
        self,
        ai_question: str,
        student_solution: str,
        topic: str
    ) -> Dict:
        """AI Examiner evaluates student solution"""
        is_predicted = True
        result = {
            "prediction": None,
            "error_message": None
        }
        try:
            output = self.llm_chain_eval_ans.predict(
                ai_question=ai_question, 
                student_solution=student_solution,
                topic=topic)
            output = output.strip()
            idx = output.find("Student grade:")
            student_grade = output[idx + 15: ].strip()
            result["prediction"] = {
                "student_grade": student_grade
            }
            
            return (is_predicted, result)
        except Exception as err:
            Logger.error("Error: {}".format(str(err)))
            is_predicted = False
            result["error_message"] = str(err)
            return (is_predicted, result)

    async def examiner_hint_motivate(
        self, 
        ai_question: str,
        student_solution: str,
        topic: str
    ) -> Dict:
        """AI Examiner hints and motivate student"""
        is_predicted = True
        result = {
            "prediction": None,
            "error_message": None
        }
        try:
            output = self.llm_chain_hint_motivate.predict(
                ai_question=ai_question, 
                student_solution=student_solution,
                topic=topic)
            output = output.strip()
            idx = output.find("Encourage student:")
            hint = output[: idx-1].strip()
            motivation = output[idx + 19: ].strip()

            result["prediction"] = {
                "hint": hint,
                "motivation": motivation
            }
            
            return (is_predicted, result)
        except Exception as err:
            Logger.error("Error: {}".format(str(err)))
            is_predicted = False
            result["error_message"] = str(err)
            return (is_predicted, result)