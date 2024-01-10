from typing import List, Dict, Optional

from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    SystemMessage
)

from .base import BaseTask


class ExampleTask(BaseTask):
    """Example task"""

    def get_dataset(self, phase: str):
        return [
            {"text": "What is 1+1?", "target": 2},
            {"text": "What is 2+2?", "target": 4},
            {"text": "What is 3+3?", "target": 6},
        ]

    def get_chain(self, generation_llm: str, feedback_llm: str, refinement_llm: str,
                  chain_name: Optional[str] = None):
        # 0. Setup
        assert chain_name is None
        initial_llm = ChatOpenAI(model_name=generation_llm)
        feedback_llm = ChatOpenAI(model_name=feedback_llm)
        refinement_llm = ChatOpenAI(model_name=refinement_llm)

        # === 1. Initial solution === #
        initial_solution_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a math question-answering assistant."),
            HumanMessagePromptTemplate.from_template("""
Question: {text}
        """.strip(), input_variables=["text"]),
        ])
        initial_solution_chain = LLMChain(
            llm=initial_llm,
            prompt=initial_solution_prompt,
            output_key="initial_solution",
        )

        # === 2. Feedback === #
        feedback_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a math question-answering assistant."),
            HumanMessagePromptTemplate.from_template("""
The following is a proposed solution to a math question. 

Question: {text}
Proposed solution: {initial_solution}

Please provide feedback on the proposed solution.
        """.strip(), input_variables=["text", "initial_solution"]),
        ])
        feedback_chain = LLMChain(llm=feedback_llm, prompt=feedback_prompt, output_key="feedback")

        # === 3. Refinement === #
        # Simulate an exchange
        refinement_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a math question-answering assistant."),
            HumanMessagePromptTemplate.from_template("""
Question: {text}
            """.strip(), input_variables=["text"]),
            AIMessagePromptTemplate.from_template("""
{initial_solution}
            """.strip(), input_variables=["initial_solution"]),
            HumanMessagePromptTemplate.from_template("""
I'm not sure about that. Here's why I think it's wrong: {feedback}
            """.strip(), input_variables=["feedback"]),
        ])
        refinement_chain = LLMChain(llm=refinement_llm, prompt=refinement_prompt, output_key="refinement")

        ilf_chain = SequentialChain(
            chains=[initial_solution_chain, feedback_chain, refinement_chain],
            input_variables=["text"],
            output_variables=["initial_solution", "feedback", "refinement"],
        )
        return ilf_chain

    def evaluate(self, phase: str, outputs: List[Dict]):
        # This is a terrible evaluation metric, but it's just an example.
        # In practice we need to parse the output and get the answer.
        dataset = self.get_dataset(phase=phase)
        scores = {"exact_match": []}
        for row, example in zip(outputs, dataset):
            exact_match = str(row["refinement"]) == str(example["target"])
            scores["exact_match"].append(exact_match)
        return {
            "exact_match": sum(scores["exact_match"]) / len(scores["exact_match"]),
        }
