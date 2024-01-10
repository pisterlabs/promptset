from typing import List, Dict, Optional
import pandas as pd

from datasets import load_dataset
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    SystemMessage
)

from .base import BaseTask
from ...utils.models import get_chat_model


class MathQATask(BaseTask):
    """MathQA task"""

    def get_dataset(self, phase: str):
        ds = load_dataset("math_qa")[phase]
        return ds

    def get_chain(self, generation_llm: str, feedback_llm: str, refinement_llm: str,
                  chain_name: Optional[str] = None):
        # 0. Setup
        assert chain_name is None
        initial_llm = get_chat_model(model_name=generation_llm)
        feedback_llm = get_chat_model(model_name=feedback_llm)
        refinement_llm = get_chat_model(model_name=refinement_llm)

        # === 1. Initial solution === #
        initial_solution_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a math question-answering assistant."),
            HumanMessagePromptTemplate.from_template("""
        The following is a math problem. Reason through the problem step-by-step, putting each separate reasoning step on a new numbered line (e.g. "Step 1. ") and finally respond with the right answer. Put the final answer letter on a single line.

        Question:
        {text}
        Options:
        {options}
            """.strip(), input_variables=["text", "options"])
        ])
        initial_solution_chain = LLMChain(llm=initial_llm, prompt=initial_solution_prompt,
                                          output_key="initial_solution")

        # === 2. Feedback === #
        ilf_feedback_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a math question-answering assistant."),
            HumanMessagePromptTemplate.from_template("""
        The following is a proposed solution to a math question. There may be an error with the solution, or it may be correct. Go through each line and indicate if that line has an error (and explain what the error is) or no error ("OK."). After that, print "REFINE" one a single line if there are errors identified, or if there are no errors, print "CORRECT".

        The output should look like:

            Step X: (Description of error)

            or 

            Step X: OK.

        for each line.

        Question:
        {text}
        Options:
        {options}

        Proposed solution:
        {initial_solution}
            """.strip(), input_variables=["text", "options", "initial_solution"])
        ])
        feedback_chain = LLMChain(llm=feedback_llm, prompt=ilf_feedback_prompt, output_key="feedback")

        # === 3. Refinement === #
        ilf_refinement_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a math question-answering assistant."),
            HumanMessagePromptTemplate.from_template("""
        You will be given a math problem with multiple-choice answers, and a proposed answer from a student. You will also be provided feedback a teacher provided on that initial solution. Based on the feedback, reason through the problem step-by-step, and finally respond with the letter corresponding to the right answer choice.

        Instruction:
        {text}
        Options:
        {options}
        Student's answer:
        {initial_solution}
        Teacher's feedback:
        {feedback}
            """.strip(), input_variables=["text", "options", "initial_solution", "feedback"])
        ])
        refinement_chain = LLMChain(llm=refinement_llm, prompt=ilf_refinement_prompt, output_key="refinement")

        ilf_chain = SequentialChain(
            chains=[initial_solution_chain, feedback_chain, refinement_chain],
            input_variables=["text", "options"],
            output_variables=["initial_solution", "feedback", "refinement"],
        )
        return ilf_chain

    def process(self, chain, example):
        return chain({"text": example["Problem"], "options": example["options"]})

    def evaluate(self, phase: str, outputs: List[Dict]):
        # This is a terrible evaluation metric, but it's just an example.
        # In practice we need to parse the output and get the answer.
        dataset = self.get_dataset(phase=phase)
        scores = {"initial_score": [], "refined_score": []}
        for row, example in zip(outputs, dataset):
            initial_solution = get_math_qa_answer(row["initial_solution"])
            refined_solution = get_math_qa_answer(row["refinement"])
            scores["initial_score"].append(example["correct"] == initial_solution)
            scores["refined_score"].append(example["correct"] == refined_solution)
        return {
            "initial_score": float(pd.Series(scores["initial_score"]).mean()),
            "refined_score": float(pd.Series(scores["refined_score"]).mean()),
        }


def get_math_qa_answer(solution):
    candidates = [
        "a)", "b)", "c)", "d)", "e)",
        "a )", "b )", "c )", "d )", "e )"
    ]
    candidates = candidates + [x.upper() for x in candidates]
    positions = {}
    for candidate in candidates:
        positions[candidate] = solution.rfind(candidate)
    srs = pd.Series(positions)
    if srs.max() == -1:
        answer = "X"
    else:
        answer = srs.idxmax()[:1].lower()
    return answer
