# -*- coding: utf-8 -*-
# @place: Pudong, Shanghai
# @file: doc_qa_evaluation.py
# @time: 2023/12/20 13:12
import os
from langchain.evaluation import load_evaluator, EvaluatorType, Criteria
from langchain.chat_models import ChatOpenAI

from utils.logger import logger
from config.config_parser import OPENAI_KEY


class DocQAEvaluation(object):
    def __init__(self, question, context, answer):
        self.question = question
        self.context = context
        self.answer = answer

    def evaluate(self):
        os.environ["OPENAI_API_KEY"] = OPENAI_KEY
        evaluation_llm = ChatOpenAI(model="gpt-4", temperature=0)

        evaluator = load_evaluator(EvaluatorType.CONTEXT_QA,
                                   criteria=Criteria.CORRECTNESS,
                                   llm=evaluation_llm,
                                   requires_reference=True)

        # evaluate
        eval_result = evaluator.evaluate_strings(
            input=self.question,
            prediction=self.answer,
            reference=self.context
        )
        logger.info(f"question: {self.question},"
                    f"context: {self.context},"
                    f"answer: {self.answer},"
                    f"evaluation: {eval_result}")

        return {"CORRECTNESS": eval_result["value"]}
