from __future__ import annotations
from typing import Any, Dict, List, Optional
import openai
import os

from langsmith import Client
from langchain.smith import RunEvalConfig, run_on_dataset

from evaluator.customer_evaluator import (
    GrammarAccuracyEvaluator,
    YAMLCorrectnessEvaluator,
    InstructionValidityEvaluator,
)
from jarvis.smartgpt.translator import Translator
from jarvis.smartgpt import gpt
from jarvis.smartgpt import initializer
from langchain.prompts.prompt import PromptTemplate

from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chat_models import ChatOpenAI
from langchain.chains.base import Chain

initializer.setup()
workspace_dir = "workspace/evaluation"
os.makedirs(workspace_dir, exist_ok=True)
os.chdir(workspace_dir)


class TranslatorMockChain(Chain):
    """
    An example of a custom chain.
    """

    input_key: str = "task_info"  #: :meta private:
    output_key: str = "output"  #: :meta private:
    jarvis_translator = Translator(gpt.GPT_4)

    class Config:
        extra = "forbid"
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        result = self.jarvis_translator.translate_to_instructions(**inputs)
        return {self.output_key: result}

    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, str]:
        result = self.jarvis_translator.translate_to_instructions(**inputs)
        return {self.output_key: result}

    @property
    def _chain_type(self) -> str:
        return "my_custom_chain"


if __name__ == "__main__":
    cot_template = """You are a Computer Science instructor specializing in the Java Virtual Machine (JVM) curriculum. With a deep understanding of JVM syntax and operational mechanics, you are now grading a quiz focusing on custom JVM syntax.
    You will be provided with a question, the student's response, and the correct answer. Your task is to evaluate the student's answer based on the true JVM syntax and determine if it is CORRECT or INCORRECT, based on the context.
    Write out in a step by step manner your reasoning to be sure that your conclusion is correct. Avoid simply stating the correct answer at the outset.

    Some JVM syntax you need to remember:
    - jvm.eval(expression): Evaluates the given expression and returns its value. This is typically used to dynamically determine arguments for other JVM instructions, especially when such arguments depend on the current state of the database or the results of previous instructions. For instance, it can be used inside the 'Loop' instruction's 'count' argument to determine the number of iterations based on a list's length stored in the database.
    - jvm.get('key_name'): returns an object of the specified key
    - jvm.set('key_name', value): sets an object to the specified key
    - jvm.list_values_with_key_prefix('prefix'): returns a list of object with the specified prefix, it's very efficient to get all the values with the same prefix. Usually work with Loop instruction together.
    - jvm.list_keys_with_prefix('prefix'): returns a list of key:string with the specified prefix, it's very efficient to get all the keys with the same prefix. Usually work with Loop instruction together.
    - In the JVM system, the suffix of a key determines the type of its value. For instance, keys ending in .list indicate that the value is a python list, while keys ending in .str signify that the value is a string. Always refer to the key's suffix to understand the expected data type.

    And some instructions you need to remember:
    1. The '<to_fill>' in instructions is a placeholder that will be replaced during execution.
    2. The types of instructions are in the range: 'WebSearch', 'FetchWebContent', 'TextCompletion', 'If', 'Loop', and 'RunPython'. And the arguments for each instruction are pre-defined.
    3. Remember that the value stored under the 'idx' key is always a number.
    4. Note: jvm.eval serves as a specialized marker, identifying portions of an expression requiring evaluation. Within any particular expression, there should only be a single segment demanding evaluation. All occurrences of jvm.get('<key>') must be nested within the scope of a jvm.eval.
    5. !IMPORTANT!: Use the syntax from the provided correct answers in the CONTEXT section as a reference for accurate syntax usage. However, different implementations (different instructions, key name, seq number, overall outcome) that still produce correct results are acceptable, so please don't compare implementations too much, minor deviations and issues can be overlooked if the student's answer achieves the same functionality or result.

    Example Format:
    QUESTION: question here
    CONTEXT: context the question is about here
    STUDENT ANSWER: student's answer here
    EXPLANATION: step by step reasoning here
    GRADE: CORRECT or INCORRECT here

    Evaluation Guide:
    1. Syntactical Accuracy: Primarily evaluate student answers based on their syntactical accuracy. Minor deviations and issues can be overlooked if the student's answer achieves the same functionality or result.
    2. Correctness of YAML: Assess the correctness of the YAML used in the student's answer. This is an important aspect to consider during the evaluation.
    3. Effectiveness of Instructions: Evaluate whether the instructions provided in the student's answer effectively achieve the desired task. The practical application and outcome are pivotal.
    While all points are important, always prioritize correctness in syntax above all.

    Begin!

    QUESTION: {query}
    CONTEXT: {context}
    STUDENT ANSWER: {result}
    EXPLANATION:"""

    COT_PROMPT = PromptTemplate(
        input_variables=["query", "context", "result"], template=cot_template
    )

    llm = gpt.OPEN_AI_MODELS_HUB[gpt.GPT_4].get_llm()

    client = Client()
    dataset_name = "Jarvis translator"

    eval_config = RunEvalConfig(
        evaluators=[
            RunEvalConfig.CoTQA(prompt=COT_PROMPT),
        ],
        custom_evaluators=[
            GrammarAccuracyEvaluator(),
            YAMLCorrectnessEvaluator(),
            InstructionValidityEvaluator(),
        ],
        eval_llm=llm,
    )

    def input_mapper(input: Dict[str, Any]):
        return {"task_info": input}

    chain_results = run_on_dataset(
        client=client,
        dataset_name=dataset_name,
        llm_or_chain_factory=lambda: TranslatorMockChain(),
        concurrency_level=1,
        evaluation=eval_config,
        num_repetitions=2,
        verbose=True,
        input_mapper=input_mapper,
    )
