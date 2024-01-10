# This file is the code used to run all multi-task batched test experiments

from src.utils.evaluation import CodeEvaluator, Evaluation
import re
from typing import Callable, List, Dict, Any, Tuple, Union, Optional, TypedDict
import os
from litellm import batch_completion
from nltk.tokenize import word_tokenize
import together
import openai
import backoff
import itertools
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, TypedDict, Optional 
from pathlib import Path
import time
import math
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import random
import pickle
from dataclasses import dataclass
from datasets import load_dataset, Dataset
from typing import Callable, List, Dict, Any, Tuple, Union, Optional, TypedDict
from enum import Enum, auto
from tqdm import tqdm
from pathlib import Path

from langchain.prompts.example_selector.semantic_similarity import (
    MaxMarginalRelevanceExampleSelector,
    SemanticSimilarityExampleSelector,
)
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings


DEBUG_NUM_QUESTIONS_WANT_ANSWER_PER_EXPERIMENT = 240


class TogetherAIGenerationParameters(TypedDict):
    model_name: str
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    logprobs: int

class OpenAIGenerationParameters(TypedDict):
    model_name: str
    temperature: float
    max_tokens: int
    frequency_penalty: float

class DebugGenerationParameters(TypedDict):
    pass

def read_api_token(token_path : str) -> str:
    # Read API token from a dedicated file
    with open(token_path, "r") as f:
        API_TOKEN = f.read().strip()
    return API_TOKEN

# TYPES AND ENUMS
ID_TYPE = Union[str, int]
EXAMPLE_FORMAT_FUNCTION_TYPE = Callable[[Dict[str, Any], Optional[int]], str]
GENERATION_PARAMETERS_TYPE = Union[
    TogetherAIGenerationParameters, 
    OpenAIGenerationParameters, 
    DebugGenerationParameters,
]

class DatasetType(Enum):
    GSM8K_HARD = auto()
    GSM8K_HARD_CoT = auto()
    COMMON_SENSE = "COMMON_SENSE"
    COMMON_SENSE_CoT = auto()
    GSM8K = "GSM8K"
    MBPP = "MBPP"
    RTE = "RTE"
    MNLI = "MNLI"

class ModelAPIType(Enum):
    TOGETHER_AI = auto()
    OPEN_AI = auto()
    DEBUG = auto()

# DICTS
DATASET_ID_KEYS = {
    DatasetType.GSM8K_HARD_CoT : ['idx'],
    DatasetType.GSM8K_HARD : ['idx'],
    DatasetType.COMMON_SENSE : ['idx'],
    DatasetType.COMMON_SENSE_CoT : ['idx'],
    DatasetType.GSM8K : ['idx'],
    DatasetType.MBPP : ['task_id'],
    DatasetType.RTE : ['idx'],
    DatasetType.MNLI : ['idx'],
}

DATASET_INPUT_KEYS = {
    DatasetType.GSM8K_HARD_CoT : ['question'],
    DatasetType.GSM8K_HARD : ['input'],
    DatasetType.COMMON_SENSE : ['question','choices'],
    DatasetType.COMMON_SENSE_CoT : ['source'],
    DatasetType.GSM8K : ['question'],
    DatasetType.MBPP : ['text','test_list'],
    DatasetType.RTE : ['sentence1', 'sentence2'],
    DatasetType.MNLI : ['premise', 'hypothesis'],
}

DATASET_LABEL_KEYS = {
    DatasetType.GSM8K_HARD_CoT : ['answer'],
    DatasetType.GSM8K_HARD : ['target'],
    DatasetType.COMMON_SENSE : ['answerKey'],
    DatasetType.COMMON_SENSE_CoT : ['rationale', 'target'],
    DatasetType.GSM8K : ['answer'],
    DatasetType.MBPP : ['code', 'test_list', 'test_setup_code', 'challenge_test_list'],
    DatasetType.RTE : ['label'],
    DatasetType.MNLI : ['label'],
}

# these are the texts that go before the Q[i] in batch prompts
# currently unused
DATASET_BATCH_INDEX_Q = {
    DatasetType.GSM8K_HARD_CoT : ['Q'],
    DatasetType.GSM8K_HARD : ['Q'],
    DatasetType.COMMON_SENSE : ['Q'],
    DatasetType.COMMON_SENSE_CoT : ['Q'],
    DatasetType.GSM8K : ['Q'],
    DatasetType.MBPP : ['Q'],
    DatasetType.RTE : ['Premise', 'Hypothesis'],
    DatasetType.MNLI : ['Premise', 'Hypothesis'],
}

# these are the texts that go before the Q[i] in batch prompts
# currently unused
DATASET_BATCH_INDEX_A = {
    DatasetType.GSM8K_HARD_CoT : ['A'],
    DatasetType.GSM8K_HARD : ['A'],
    DatasetType.COMMON_SENSE : ['A'],
    DatasetType.COMMON_SENSE_CoT : ['A'],
    DatasetType.GSM8K : ['A'],
    DatasetType.MBPP : ['A'],
    DatasetType.RTE : ['A'],
    DatasetType.MNLI : ['A'],
}

class ExampleSelectionType(Enum):
    RANDOM = auto()
    SEMANTIC = auto()
    LEXICAL = auto()
    MAX_MARGINAL_RELEVANCE = auto()

@dataclass
class DatasetConfig:
    dataset : DatasetType
    # could be the name of a dataset, or a list of strings that specify the path to a dataset 
    # e.g. 'mbpp' vs ['mbpp', 'sanitized']
    hf_dataset_path: Optional[Union[str, List[str]]] = None
    task_name_for_CoT_filter: Optional[str] = None
    split_name: Optional[str] = None
    # can also choose to load a dataset from a json file
    local_path: Optional[Path] = None

    def __post_init__(self):
        # validate the config
        self.validate()

    def validate(self):
        match (self.hf_dataset_path is not None, self.split_name is not None, self.local_path is not None):
            case (True, False, _) | (False, True, _):
                raise ValueError("Must either both or neither specify a huggingface dataset path and a split name")
            case (True, True , True):
                raise ValueError("Cannot specify both a local path and a huggingface dataset path")
            case(False, False, False):
                raise ValueError("Must specify either a local path or a huggingface dataset path")
            case _: pass
        
        if self.local_path is not None:
            if not self.local_path.exists():
                raise ValueError(f"Local path {self.local_path} does not exist")
            if not self.local_path.is_file():
                raise ValueError(f"Local path {self.local_path} is not a file")
            if not self.local_path.suffix == '.json':
                raise ValueError(f"Local path {self.local_path} is not a json file")

@dataclass
class MultiTaskBatchPromptingDebugConfig:
    truncate_examples : bool = False,
    truncate_batch_queries : bool = False
    save_batched_model_inputs : Optional[Path] = None
    save_batched_model_outputs : Optional[Path] = None

@dataclass
class MultiTaskBatchPromptingExperimentConfig:
    # can either load a dataset from huggingface or from a local json file
    questions_dataset_config : Dict[DatasetType, DatasetConfig]
    task_descriptions: Dict[DatasetType, str]
    objective_instructions: str
    io_instructions: str
    k_shot: int
    batch_size: int
    question_format: Dict[DatasetType, EXAMPLE_FORMAT_FUNCTION_TYPE]
    model_api: ModelAPIType
    generation_params: GENERATION_PARAMETERS_TYPE
    debug : Optional[MultiTaskBatchPromptingDebugConfig] = None



# a list of examples (dicts with featues) of different types:
MT_EXAMPLE_TYPE = Tuple[DatasetType, Dict[str, Any]]
MT_ID_TYPE = Tuple[DatasetType, ID_TYPE]


class MultiTaskBatchPromptExperiment:
    def __init__(
            self,
            config: MultiTaskBatchPromptingExperimentConfig,
    ):
        self.config = config

        self.questions = {
            dataset_key : self.load_dataset(dataset_config)
            for dataset_key, dataset_config in self.config.questions_dataset_config.items()
        }

        # must add an index column to gsm8k
        self.debug = self.config.debug
        self.batch_prompt_template = MultiTaskBatchPromptTemplate(
            datasets=list(self.questions.keys()),
            objective_instructions=self.config.objective_instructions,
            task_descriptions=self.config.task_descriptions,
            io_instructions=self.config.io_instructions,
            num_questions=self.config.batch_size,
            question_format=self.config.question_format,
            debug=self.config.debug,
        )

    def load_dataset(self, dataset_config: DatasetConfig) -> Dataset:
        if dataset_config.local_path is not None:
            # load locally
            dataset = load_dataset(
                'json', 
                data_files=dataset_config.local_path,
                split='train', # loading from file makes a dataset with only train split
            )
        else:
            # load from huggingface
            dataset = load_dataset(
                *dataset_config.hf_dataset_path,
                split=dataset_config.split_name,
            )
            if dataset_config.task_name_for_CoT_filter != None:
                dataset = dataset.filter(lambda example: example['task'] == dataset_config.task_name_for_CoT_filter)
                if dataset_config.task_name_for_CoT_filter =="rte":
                    dataset = dataset.filter(lambda example: "Question with options: can we draw the following hypothesis from the context?" in example['source'])
                if dataset_config.task_name_for_CoT_filter =="mnli":
                    dataset = dataset.filter(lambda example: "Premise: " in example['source'] and "Hypothesis: " in example['source'])
        # add an index column to gsm8k
        match dataset_config.dataset:
            case DatasetType.COMMON_SENSE | DatasetType.GSM8K:
                dataset = dataset.add_column('idx', list(range(len(dataset))))
        return dataset
    
    # The params here are for LLAMA 2 with togetherai
    def batch_query_model(
            self, 
            model_inputs: List[Tuple[List[MT_ID_TYPE], str]]
        ) -> List[Tuple[List[MT_ID_TYPE], str]]:

            messages = [[{"role": "user", "content": i[1]}] for i in model_inputs]
            attempt_cnt = 0
            max_attempts = 10 
            model_query_batch_size = 2
            tokens_per_message = [len(word_tokenize(model_input[1])) for model_input in model_inputs]
            # Note: *1.2 is because output tokens will count towards our minutely token limit
            tokens_per_batch_completion = [sum(tokens_per_message[i:i+model_query_batch_size])*1.2 for i in range(0, len(messages), model_query_batch_size)]
            token_rate_limit = 40_000 # This is for GPT 3.5 Turbo token limit per minute
            results = []
            
            model_name = self.config.generation_params["model_name"]
            generation_params = {
                k : v for k, v in self.config.generation_params.items() if k != "model_name"
            }

            message_sublists = [messages[i:i+model_query_batch_size] for i in range(0, len(messages), model_query_batch_size)]
            for batched_messages, tokens_in_batch in zip(message_sublists, tokens_per_batch_completion):
                while attempt_cnt < max_attempts:
                    with ThreadPoolExecutor() as executor:
                        future = executor.submit(batch_completion, model=model_name, messages = batched_messages, **generation_params,)
                        try:
                            batch_exectution_start_time = time.time()
                            curr_results = future.result(timeout=60)
                            batch_exectution_end_time = time.time()
                            batch_execution_time = batch_exectution_end_time - batch_exectution_start_time
                            results.extend(curr_results)
                            

                            # Calculate the time needed to wait based on tokens processed and token rate limit
                            tokens_per_second = token_rate_limit / 60
                            expected_time_for_tokens = tokens_in_batch / tokens_per_second
                            sleep_time = max(0, expected_time_for_tokens - batch_execution_time)
                            print(f"batch_execution_time: {batch_execution_time}, tokens_per_second: {tokens_per_second}, expected_time_for_tokens: {expected_time_for_tokens}, sleep_time: {sleep_time}")
                            time.sleep(sleep_time)

                            break
                        except TimeoutError:
                            attempt_cnt += 1
                            print(f"Timeout error occurred. Retrying attempt {attempt_cnt}...")
                            time.sleep(20*attempt_cnt)  # Add a short delay before retrying
                        except Exception as e:
                            attempt_cnt += 1
                            print(f"Error {str(e)} occurred. Retrying attempt {attempt_cnt}...")
                            time.sleep(20*attempt_cnt)  # Add a short delay before retrying
                if attempt_cnt == max_attempts:
                    curr_results = ["" for i in range(len(batched_messages))]
                    results.extend(curr_results)
            ids = [i[0] for i in model_inputs]
            return list(zip(ids, results))  

    # returns a list of batches of questions and their associated ids
    def batch_questions(self) -> List[Tuple[List[MT_ID_TYPE], List[MT_EXAMPLE_TYPE]]]:
        all_questions : Dict[DatasetType, MT_EXAMPLE_TYPE] = {}
        ids : Dict[DatasetType, MT_ID_TYPE] = {}
        for dataset_key, dataset in self.questions.items():
            all_questions[dataset_key] = []
            ids[dataset_key] = []
            for example in dataset:
                all_questions[dataset_key].append((dataset_key, example))
                example_id = example[DATASET_ID_KEYS[dataset_key][0]]
                ids[dataset_key].append((dataset_key, example_id))
        
        # batch the questions
        batched_questions  = []
        
        # the largest number divisible by both 6 and 8 that is less than 277, the number of examples in RTE
        truncation = 240
        
        randomized_indices : Dict[DatasetType, List[Tuple[DatasetType, int]]] = {}
        for dataset_key, dataset in self.questions.items():
            indices : List[Tuple[DatasetType, int]] = [(dataset_key, i) for i in range(len(dataset))]
            random.shuffle(indices)
            randomized_indices[dataset_key] = indices[:truncation]

        # by cycling one from each dataset at a time, we ensure that the batches are balanced 
        # (i.e. have at least one from each dataset)
        one_from_each = list(zip(*randomized_indices.values()))
        all_indices = list(itertools.chain(*one_from_each))

        # will contain batches of (dataset, index) tuples
        batched_indices : List[List[Tuple[DatasetType, int]]] = []
        for i in range(0, len(all_indices), self.config.batch_size):
            batch = all_indices[i:i+self.config.batch_size]
            random.shuffle(batch)
            batched_indices.append(batch)
        
        # now we have a list of batches of (dataset, index) tuples
        batched_ids : List[List[MT_ID_TYPE]] = []
        batched_examples : List[List[MT_EXAMPLE_TYPE]] = []
        for batch in batched_indices:
            batched_ids.append([ids[ds][i] for (ds, i) in batch])
            batched_examples.append([all_questions[ds][i] for (ds, i) in batch])
        
        batched_questions : List[
            Tuple[List[MT_ID_TYPE],
            List[MT_EXAMPLE_TYPE]]
        ] = list(zip(batched_ids, batched_examples))

        return batched_questions
    
    # gives a dict from (dataset_type, question_id) to the datapoint dict (which has the answers)
    def answers_from_batched_questions(
        self, 
        batched_questions: List[Tuple[List[MT_ID_TYPE], List[MT_EXAMPLE_TYPE]]]
    ) -> Dict[MT_ID_TYPE, Dict[str, Any]]:
        answers :  Dict[ID_TYPE, Dict[str, Any]] = {
            (dataset_type, question_id) : question
            for (ids, questions) in batched_questions
            for ((dataset_type, question_id), (_, question)) in zip(ids, questions)
        }
        return answers


    def execute(self):
        """
        X Load Dataset
        X Generate set of model inputs (using dataset + config + FlexiblePromptTemplate)
        X query model for each input (save raw outputs to a file somewhere)
        """
        # splits self.questions into batches that are lists of individual dictionaries along with their ids
        batched_questions: List[Tuple[List[MT_ID_TYPE], List[MT_EXAMPLE_TYPE]]] = self.batch_questions()
        answers_dict : Dict[MT_ID_TYPE, Dict[str, Any]] = self.answers_from_batched_questions(batched_questions)

        # generate prompts for each batch
        batched_model_inputs : List[Tuple[List[ID_TYPE], str]] = [
            (ids, self.batch_prompt_template.generate_prompt(batch))
            for (ids, batch) in batched_questions
        ]
        # for debug purposes
        if self.debug:
            total_number_examples_wanted = DEBUG_NUM_QUESTIONS_WANT_ANSWER_PER_EXPERIMENT
            needed_llm_calls = total_number_examples_wanted/self.config.batch_size
            if needed_llm_calls.is_integer():
                needed_llm_calls = int(needed_llm_calls)
            else:
                needed_llm_calls = math.ceil(needed_llm_calls)
            if needed_llm_calls == 0:
                needed_llm_calls = 1
            needed_llm_calls = min(needed_llm_calls, len(batched_model_inputs))
            batched_model_inputs = batched_model_inputs[:needed_llm_calls]

        batched_model_outputs = self.batch_query_model(batched_model_inputs)
        if self.debug != None and batched_model_inputs:
            if self.debug.save_batched_model_inputs:
                pickle.dump((batched_model_inputs), open(self.debug.save_batched_model_inputs, 'wb'))
            if self.debug.save_batched_model_outputs:
                pickle.dump((batched_model_outputs), open(self.debug.save_batched_model_outputs, 'wb'))

        return (batched_model_inputs, batched_model_outputs, answers_dict)

class MultiTaskBatchPromptTemplate:
    """
    MultiTaskBatchPromptTemplate is a class that generates prompts for a batch of examples that can have a mix of tasks.
    It is used to generate prompts for the multi-task k-shot experiment.

    prompts have the following:

    Objective:

    Task and Token(?) Description(s):

    Unified Input/Output Format Instructions:

    Examples:

    Batched Questions:
    """
    def __init__(
            self,
            datasets: List[DatasetType],
            objective_instructions: str,
            task_descriptions: Dict[DatasetType, str],
            io_instructions: str,
            num_questions: int,
            question_format: Dict[DatasetType, Callable[[Dict[str, Any], Optional[int]], str]],
            debug: Optional[MultiTaskBatchPromptingDebugConfig] = None,
    ):
        self.datasets = datasets
        self.objective_instructions = objective_instructions
        self.task_descriptions = task_descriptions
        self.io_instructions = io_instructions
        self.num_questions = num_questions
        self.question_format = question_format
        self.debug = debug

    def generate_prompt(self, batch: List[MT_EXAMPLE_TYPE]) -> str:
        """
        Generates a prompt for a batch of examples
        """
        objective_instructions = self.objective_instructions

        task_descriptions = "Task Descriptions:\n" + "\n".join([
            self.task_descriptions[task]
            for task in self.datasets
        ])
        
        io_instructions = self.io_instructions

        batched_questions = "\n".join([
            f"{self.question_format[dataset](example, i)}"
            for i, (dataset, example) in enumerate(batch)
        ])

        prompt = '''\
{objective_instructions}

{task_and_token_descriptions}

{io_instructions}

Batched Questions to Answer:
{batched_questions}'''.format(
            objective_instructions=objective_instructions.format(batch_size=self.num_questions),
            task_and_token_descriptions=task_descriptions,
            io_instructions=io_instructions,
            batched_questions=batched_questions,
        )
        return prompt

DATASET_QUESTIONS_CONFIGS = {
    DatasetType.RTE : DatasetConfig(
        dataset=DatasetType.RTE,
        hf_dataset_path=['glue', 'rte'],
        split_name='validation',
    ),
    DatasetType.COMMON_SENSE : DatasetConfig(
        dataset=DatasetType.COMMON_SENSE,
        hf_dataset_path=['commonsense_qa'],
        split_name='validation',
    ),
    DatasetType.GSM8K : DatasetConfig(
        dataset=DatasetType.GSM8K,
        hf_dataset_path=['gsm8k', 'main'],
        split_name='test',
    ),
    DatasetType.MNLI : DatasetConfig(
        dataset=DatasetType.MNLI,
        hf_dataset_path=['glue', 'mnli'],
        split_name='validation_matched',
    ),
}

DATASET_TASK_DESCRIPTIONS = {
    DatasetType.COMMON_SENSE : '''\
COMMON_SENSE: Instruction - our task is to solve a set of multiple-choice questions from the CommonsenseQA dataset in a batch. CommonsenseQA is a new multiple-choice question answering dataset that requires different types of commonsense knowledge to predict the correct answers . You will be given {{batch_size}} questions each time, as input. These questions are designed to test your ability to answer queries that often require contextual understanding and general world knowledge. Your goal is to select the letter corresponding to the most appropriate answer among five options labeled 'a', 'b', 'c', 'd', and 'e' for each question in the batch.
COMMON_SENSE: Method - Use your expertise in NLP and contextual understanding to perform a sequence of logical evaluations for solving these questions.
COMMON_SENSE: Intermediate Reasoning - Include all the steps you took to arrive at your answer. This could include identifying key phrases, contradictions, or logical connections that led you to choose a particular option.
COMMON_SENSE: Output Meaning - Select the most appropriate answer and output the letter after "The answer is" with the corresponding letter.''',
    DatasetType.GSM8K : '''\
GSM8K: Instruction - Your task is to solve a set of math questions in a batch.
GSM8K: Method - Use basic arithmetic operations to perform a sequence of calculations for solving these questions.
GSM8K: Intermediate Reasoning -  Each question in the batch will require you to perform between 2 and 8 steps to arrive at the final answer.
GSM8K: Output Meaning - Each answer is an integer that is the answer to the question.''',
    DatasetType.RTE : '''\
RTE: Instruction - Your task is to solve a set of recognizing textual entailment (RTE) questions in a batch. You will be given {{batch_size}} sentence pairs from the Textual Entailment Recognition dataset each time, as input. Your goal is to classify each sentence pair into two classes.
RTE: Method - Use your expertise in NLP and sentence pair relationship annotation to perform a sequence of logical evaluations for solving these questions.
RTE: Intermediate Reasoning - Include all the steps you took to evaluate the relationship between the Premise and Hypothesis. This could include identifying key phrases, contradictions, or logical connections.
RTE: Output Meaning - An answer of 0 signifies entailment between the Hypothesis and Premise, while 1 signifies non-entailment.''', 
    DatasetType.MNLI : '''\
MNLI: Instruction - Your task is to solve a set of MultiNLI (MNLI) questions in a batch.  You will be given premise-hypothesis pairs from the MNLI dataset as input. Your goal is to classify each pair into one of three classes: entailment, neutral, or contradiction.
MNLI: Method - Use your expertise in NLP and sentence pair relationship annotation to perform a sequence of logical evaluations relationship between each Premise and Hypothesis pair. Given a premise sentence and a hypothesis sentence, the task is to predict whether the premise entails the hypothesis (entailment), contradicts the hypothesis (contradiction), or neither (neutral).
MNLI: Intermediate Reasoning - Include all the steps you took to evaluate the relationship between the Premise and Hypothesis. This could include identifying key phrases, contradictions, or logical connections.
MNLI: Output Meaning - An answer of 0 means the premise entails the hypothesis, indicating that if the premise is true, the hypothesis must also be true. In this case, the information in the hypothesis is a logical subset of the information in the premise.
An answer of 1 means the relationship between the premise and the hypothesis is neutral, suggesting that the truth of the premise neither guarantees nor contradicts the truth of the hypothesis. The hypothesis could be either true or false regardless of the premise's truth value.
An answer of 2 means the premise contradicts the hypothesis, implying that both cannot be true at the same time. If the premise is true, the hypothesis must necessarily be false, and vice versa.'''
}

def gsm8k_question_format(example, i):
    example_question_format = f"Q[{i}][GSM8K] {example['question']}"
    return example_question_format

def commonsense_question_format(example, i):
        prompt = f"Q[{i}][COMMON_SENSE] {example['question']}\n"
        prompt += "\n".join(
            [f"{label}: {text}" for label, text in zip(example["choices"]["label"], example["choices"]["text"])]
        )
        return prompt

def rte_question_format(example, i): 
    return f"Q[{i}][RTE]\nP: {example['sentence1']}\nH: {example['sentence2']}"

def mnli_question_format(example, i): 
    return f"Q[{i}][MNLI]\nP: {example['premise']}\nH: {example['hypothesis']}"

QUESTION_FORMAT_FUNCTIONS = {
    DatasetType.COMMON_SENSE : commonsense_question_format,
    DatasetType.GSM8K : gsm8k_question_format,
    DatasetType.RTE : rte_question_format,
    DatasetType.MNLI : mnli_question_format,
}

IO_INSTRUCTIONS = """\
#### Instructions:

1. **Intermediate Reasoning**: Include all the steps you took to evaluate the relationship between the Premise and Hypothesis. This could include identifying key phrases, contradictions, or logical connections.
2. **Batch Size**: You must provide an answer for each question in the batch, ensuring that the number of answers you provide exactly matches the specified 4.
3. **Handling Ambiguities**: Answer every question even if you are unsure about the answer.

Unified Input Format:
The batch will contain questions each tagged with a unique index and a task code that specifies the domain of the task, formatted as follows:

Q[index][task_code]: {{Question_Text}}

The index is a zero-based number indicating the question’s position in the batch.

The task_code identifies the task domain for the question and is one of the following: ["RTE","GSM8K", "COMMON_SENSE", "MNLI"].

{{Question_Text}} is the content of the question or sentence pair to be evaluated.

#### Output Format:
- You must adhere to the following format rigorously for each answer:
A[index]: {{Intermediate_Reasoning}}; The answer is {{Answer_Integer}}
- `index`: This is the index of the question you are answering. It must be prefixed with 'A' and enclosed in square brackets.
- `{{Intermediate_Reasoning}}`: This is where you provide all the intermediate steps that led you to the final answer.
- `{{Answer_Integer}}`: This is the final integer answer to each question.

The phrase 'The answer is' must directly precede each integer answer and come after the intermediate reasoning, separated by a semicolon. Ensure you output A[index] for each question before outputting {{Intermediate_Reasoning}}; The answer is {{Answer_Integer}}. Please adhere strictly to these guidelines to ensure the entire output is in the desired format. Output all answers, ensuring that exactly {batch_size} answers are provided in our desired format. Do not include ANY reasoning after "The answer is", just the designated answer symbol."""

OBJECTIVE_INSTRUCTIONS = "Objective: Your task is to solve a variety of questions across multiple domains in a single batch operation. You will be given a number of questions, each associated with a specific task domain, and your goal is to answer each question according to its domain while adhering to the desired output format. The total number of questions in the batch to answer is defined as batch_size = {batch_size}."

oai_gen_params = OpenAIGenerationParameters(
    model_name='gpt-3.5-turbo-16k',
    temperature=0.0,
    max_tokens=None,
    frequency_penalty=1.0,
)
togetherai_gen_params =  TogetherAIGenerationParameters(
model_name='together_ai/togethercomputer/llama-2-13b-chat',
temperature = 0,
max_tokens=1600, # 4096 max but some tokens must be used for input
frequency_penalty=1.0,)
                       
                    
def split_answers(text):
    # Initialize an empty list to store the individual answers
    answer_list = []
    
    idx = 0
    while True:
        # Dynamic regular expression pattern to match "A[idx]" or "[Aidx]"
        pattern = r"(?:A\[" + str(idx) + r"\]|\[A" + str(idx) + r"\])"
        
        # Find all matches of the pattern in the text using finditer
        matches = [m for m in re.finditer(pattern, text)]
        
        # If no matches are found, break the loop
        if not matches:
            break
        
        # Get the first match for the current index
        first_match = matches[0]
        
        # Dynamic regular expression pattern to match "A[idx+1]" or "[Aidx+1]"
        next_pattern = r"(?:A\[" + str(idx + 1) + r"\]|\[A" + str(idx + 1) + r"\])"
        
        # Find all matches of the next pattern in the text using finditer
        next_matches = [m for m in re.finditer(next_pattern, text)]
        
        # If matches for the next index are found, get the last match
        if next_matches:
            next_match = next_matches[0]
            # Slice the text from the first match's start position to the next match's start position
            answer_list.append(text[first_match.start():next_match.start()])
        else:
            # If no next match is found, slice from the first match's start position to the end of the text
            answer_list.append(text[first_match.start():])
        
        # Increment the index for the next iteration
        idx += 1
    
    return answer_list
def extract_last_letter(text):
    # Define the regex pattern to match 'The answer is ' followed by a single letter (A-E or a-e)
    # The \b ensures that the letter is a word boundary, so nothing comes immediately after it.
    pattern = r'he answer is .*?([A-Ea-e])\b'
    
    # Use re.findall() to find all occurrences of the pattern
    matches = re.findall(pattern, text)
    
    if not matches:
        pattern = r'nswer is .*?([A-Ea-e])\b'
        # Use re.findall() to find all occurrences of the pattern
        matches = re.findall(pattern, text)
        if not matches:
            return None
    
    # Take the last occurrence
    last_match = matches[-1]
    
    # Convert to uppercase if it's not
    last_match = last_match.upper()
    
    return last_match

def extract_last_integer(text):
    # Regex pattern to match the desired number format
    pattern = r"answer is\s*[\[\$\'_*\"]*(-?\d{1,3}(?:,\d{3})*|\d+)"


    # Find all matches
    matches = re.findall(pattern, text)
    if not matches:
        pattern = r"answer is\s*.*?(-?\d{1,3}(?:,\d{3})*|\d+)"
        matches = re.findall(pattern, text)
        if not matches:
            return None
    # Remove commas and other formatting to clean the matched numbers
    cleaned_matches = [match.replace(',', '') for match in matches]
    
    # Adjust the matches to retain the negative sign if present
    final_matches = [f"-{match}" if text.find("-" + match) != -1 else match for match in cleaned_matches]
    
    answer = int(final_matches[-1])
    return answer

# This variant only index to predictions for answers of the desired task name
def get_index_to_pred(batched_model_inputs, batched_model_outputs, desired_task_name):
    index_to_pred = {}
    for batch_input, batch in zip(batched_model_inputs, batched_model_outputs):
        indices = [batch_input[0][i][1] for i in range(len(batch_input[0]))]
        # LLM_output = batch
        LLM_output = batch[1]["choices"][0]['message']['content']
        task_names = [batch[0][i][0].name for i in range(len(batch[0]))]
        text_split_by_batch = split_answers(LLM_output)
        for answer, task, index in zip(text_split_by_batch, task_names, indices):
            if task != desired_task_name:
                continue
            if task == "COMMON_SENSE":
                answer_parsed = extract_last_letter(answer)
            else:
                answer_parsed = extract_last_integer(answer)
            index_to_pred[index] = answer_parsed

        if len(indices) > len(text_split_by_batch):
            for index, task in zip(indices[len(text_split_by_batch):], task_names[len(text_split_by_batch):]):
                if task != desired_task_name:
                    continue
                index_to_pred[index] = None

    return index_to_pred
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import numpy as np
def get_stats(y_pred, y_true, answer_type):
    results = {}
    try:        
        def _calculate_f1(y_pred, y_true):
            if len(set(y_true)) > 2:  # More than two classes indicates multiclass classification
                return f1_score(y_true, y_pred, average='weighted')
            else:
                return f1_score(y_true, y_pred, average='binary')
                    
        # Calculate the cannot_parse_proportion
        cannot_parse_count = sum(pred is None for pred in y_pred)
        total_predictions = len(y_pred)
        cannot_parse_proportion = cannot_parse_count / total_predictions
        
        # Add the cannot_parse_proportion to the results
        results['cannot_parse_proportion'] = cannot_parse_proportion
        
        # Set1: Skip all None values
        valid_indices = [i for i, pred in enumerate(y_pred) if pred is not None]
        valid_y_pred = np.array([y_pred[i] for i in valid_indices])
        valid_y_true = np.array([y_true[i] for i in valid_indices])
        results['none_is_skipped'] = {
            'Accuracy': accuracy_score(valid_y_pred, valid_y_true)
        }
        if answer_type in ['binary', 'categorical']:
            results['none_is_skipped']['F1'] = _calculate_f1(valid_y_pred, valid_y_true)
            if answer_type == "binary":
                results['none_is_skipped']["confusion_matrix"] = confusion_matrix(valid_y_pred, valid_y_true)
                # results['none_is_skipped']['Sensitivity'] = calculate_sensitivity(valid_y_pred, valid_y_true)

        # Set2: Count None as wrong
        wrong_label = next(iter(set(y_true) - set(y_pred)), next(iter(set(y_true))))
        y_pred_set2 = [pred if pred is not None else wrong_label for pred in y_pred]
        results['none_is_wrong'] = {
            'Accuracy': accuracy_score(y_pred_set2, y_true)
        }
        if answer_type in ['binary', 'categorical']:
            results['none_is_wrong']['F1'] = _calculate_f1(y_pred_set2, y_true)
            if answer_type == "binary":
                results['none_is_wrong']["confusion_matrix"] = confusion_matrix(y_pred_set2, y_true)
                # results['none_is_wrong']['Sensitivity'] =calculate_sensitivity(y_pred_set2, y_true)

        # Set3: Guess randomly if None
        y_pred_set3 = [pred if pred is not None else random.choice(list(set(y_true))) for pred in y_pred]
        results['none_is_random'] = {
            'Accuracy': accuracy_score(y_pred_set3, y_true)
        }
        if answer_type in ['binary', 'categorical']:
            results['none_is_random']['F1'] = _calculate_f1(y_pred_set3, y_true)
            if answer_type == "binary":
                results['none_is_random']["confusion_matrix"]  = confusion_matrix(y_pred_set3, y_true)
                # results['none_is_random']['Sensitivity'] = calculate_sensitivity(y_pred_set3, y_true)
    except Exception as e:
        print(e)
    return results

def convert_to_int(input_str: str) -> int:
    # Step 1: Remove unnecessary characters like spaces and commas
    cleaned_str = re.sub(r"[^\d.-]", "", input_str)
    
    # Step 2: Convert the cleaned string to float
    float_val = float(cleaned_str)
    
    # Step 3: Convert the float to integer
    int_val = int(float_val)
    
    return int_val

def get_index_to_ground_truth(answers_dict, desired_task_name):
    index_to_ground_truth = {}
    for answer_information, answer in answers_dict.items():
        task_name = answer_information[0].name
        if task_name != desired_task_name:
            continue
        answer_index = answer_information[1]
        if task_name == "MBPP":
            raise NotImplementedError()
        elif task_name == "GSM8K":
            index_to_ground_truth[answer_index] = convert_to_int(answer["answer"].split("####")[-1])
        elif task_name == "GSM8K_HARD":
            raise NotImplementedError()
        elif task_name == "COMMON_SENSE":
            index_to_ground_truth[answer_index] = answer['answerKey']
        elif task_name in ["RTE", "MNLI"]:
            index_to_ground_truth[answer_index] = int(answer["label"])
        else:
            raise ValueError("Task name not recognized.")
    return index_to_ground_truth

def get_ordered_lists(index_to_pred: dict, index_to_ground_truth: dict) -> (list, list):
    # Initialize empty lists for predictions and ground truth values.
    pred = []
    ground_truth = []
    
    # Ensure both dictionaries have the same keys, otherwise raise an exception.
    if set(index_to_pred.keys()) > set(index_to_ground_truth.keys()):
        raise ValueError("The keys in both dictionaries should match.")
    
    # Sort the keys to ensure the values are ordered.
    sorted_keys = sorted(index_to_pred.keys())
    
    # Populate the 'pred' list with prediction values in sorted order of keys.
    for key in sorted_keys:
        pred.append(index_to_pred[key])
        
    # Populate the 'ground_truth' list with ground truth values in sorted order of keys.
    for key in sorted_keys:
        ground_truth.append(index_to_ground_truth[key])
        
    return pred, ground_truth

def get_pred_ground_truth(batched_model_inputs, batched_model_outputs, answers_dict, task_name):
    index_to_pred = get_index_to_pred(batched_model_inputs, batched_model_outputs, task_name)
    index_to_ground_truth = get_index_to_ground_truth(answers_dict, task_name)
    pred, ground_truth = get_ordered_lists(index_to_pred, index_to_ground_truth)
    return pred, ground_truth

def run_experiments():
    batch_size = 4
    dataset_types = [
        DatasetType.COMMON_SENSE,
        DatasetType.GSM8K,
        DatasetType.RTE,
        DatasetType.MNLI,
    ]
    
    dataset_combinations = list(itertools.chain(*list(itertools.combinations(dataset_types, i) for i in [1, 2,3,4])))

    dataset_combination_to_output = {}
    dataset_combination_to_stats = {}

    for dataset_combination in dataset_combinations:
        config = MultiTaskBatchPromptingExperimentConfig(
            questions_dataset_config={
                dataset_type: DATASET_QUESTIONS_CONFIGS[dataset_type]
                for dataset_type in dataset_combination
            },
            task_descriptions={
                dataset_type: DATASET_TASK_DESCRIPTIONS[dataset_type]
                for dataset_type in dataset_combination
            },
            objective_instructions=OBJECTIVE_INSTRUCTIONS,
            io_instructions=IO_INSTRUCTIONS,
            k_shot=0,
            batch_size=batch_size,
            question_format={
                dataset_type: QUESTION_FORMAT_FUNCTIONS[dataset_type]
                for dataset_type in dataset_combination
            },
            # model_api=ModelAPIType.OPEN_AI,
            # generation_params=oai_gen_params,
            model_api=ModelAPIType.TOGETHER_AI,
            generation_params=togetherai_gen_params,
        #     debug = MultiTaskBatchPromptingDebugConfig(
        #     truncate_examples=True,
        #     truncate_batch_queries=True,
        #     save_batched_model_inputs=Path("mt_batched_model_inputs.pkl"),
        #     save_batched_model_outputs=Path("mt_batched_model_outputs.pkl"),
        )
        experiment = MultiTaskBatchPromptExperiment(config)
        os.environ['TOGETHERAI_API_KEY'] = read_api_token(Path("data/imported/together_ai_token.txt"))
        os.environ['OPENAI_API_KEY'] = read_api_token(Path("data/imported/open_ai_token.txt"))
        output = experiment.execute()
        dataset_combination_to_output[dataset_combination] = output
        (batched_model_inputs, batched_model_outputs, answers_dict) = output
        task_names = ["RTE", "GSM8K", "MNLI", "COMMON_SENSE"]
        task_name_to_stats = {}
        for task_name in task_names:
            pred, ground_truth = get_pred_ground_truth(batched_model_inputs, batched_model_outputs, answers_dict, task_name)
            stat = get_stats(y_pred=pred, y_true=ground_truth, answer_type = None)
            task_name_to_stats[task_name] = stat
        dataset_combination_to_stats[dataset_combination] = task_name_to_stats

    with open("dataset_combination_to_output_llama_13B", 'wb') as f:
        pickle.dump(dataset_combination_to_output, f)
    with open("dataset_combination_to_stats_llama_13B", 'wb') as f:
        pickle.dump(dataset_combination_to_stats, f)
    print("END OF EXPERIMENT FOR MULTI TASK")


if __name__ == "__main__":
    run_experiments()