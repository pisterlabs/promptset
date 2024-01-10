# This file is the code used to run all single-task batched test experiments
import warnings
from dill._dill import PicklingWarning
from src.utils.evaluation import CodeEvaluator, Evaluation
import re
from typing import Callable, List, Dict, Any, Tuple, Union, Optional, TypedDict
import os
from litellm import batch_completion
import together
import openai
import backoff
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, TypedDict, Optional 
from pathlib import Path
import time
import math
from src.experiments.k_shot_experiment_configs import task_description_rte, task_description_COMMON_SENSE, task_description_MNLI, task_description_GSM8K
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import random

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

class LanguageModel:
    def __init__(self):
        raise NotImplementedError

    def query(self, prompt : str) -> str:
        raise NotImplementedError

class TogetherAIModel(LanguageModel):
    """ 
    A wrapper class for the Together AI API.
    
    NOTE: do NOT specify any default values for model/generation parameters.
    I'd like to keep the configurations separate from the code as much as possible
    so we have good documentation/reproducibility of the experiments without having
    to dig into the code/commit history.

    Attributes:
        api_token (str): The API token for the Together AI API.
        model_name (str): The name of the model to use.
            Llama2-70B is togethercomputer/llama-2-70b, for example.
        generation_params (dict): The parameters to use for generation.
            - max_tokens (int): The maximum number of tokens to generate.
            - temperature (float): The temperature to use for generation.
            - top_p (float): The top_p to use for generation.
            - top_k (int): The top_k to use for generation.
            - repetition_penalty (float): The repetition penalty to use for generation.
            - logprobs (int): The number of logprobs to return.
    """
    def __init__(
        self,
        api_token : str,
        model_name : str,
        generation_params : dict,
    ):
        together.api_key = api_token
        self.api_token = api_token
        self.model_name = model_name
        self.generation_params = generation_params

    def __repr__(self):
        return f"TogetherAIModel(model_name={self.model_name}, generation_params={self.generation_params})"

    @backoff.on_exception(backoff.expo, Exception, max_tries=10)
    def query(self, prompt : str) -> str:
        raise ""
        if "model_name" in self.generation_params:
            model_name = self.generation_params["model_name"]
        for attempt in range(10):  # Try up to 10 times
            try:
                response = together.Complete.create(
                    prompt=prompt,
                    model=self.model_name,
                    temperature=0,
                    max_tokens=2048,
                    # frequency_penalty=1.0
                    # **self.generation_params,
                )
                return response["output"]["choices"][0]["text"]
            except Exception as e:
                print(f"An error occurred: {e}. Retrying...")
                time.sleep(1)  # Sleep for 1 second before retrying
        z = ""
        return z
    

class OpenAIModel(LanguageModel):
    def __init__(
        self,
        api_token,
        model_name,
        generation_params,
    ):
        openai.api_key = api_token
        self.api_token = api_token
        self.model_name = model_name
        self.generation_params = {key : value for key, value in generation_params.items() if key != 'model_name'}
        


    def __repr__(self):
        return f"OpenAIModel(model_name={self.model_name}, generation_params={self.generation_params})"
        
    # @backoff.on_exception(backoff.expo, openai.error.RateLimitError, max_tries=10)
    def query(self, prompt : str):
        raise ""
        message = [{"role": "user", "content": prompt}]
        # Estimate the number of tokens used
        estimated_tokens = len(prompt.split()) * 3
        # Set the rate limits for different models
        rate_limit = 10_000 if "gpt-4" in self.model_name else 90_000
        
        try_cnt = 0
        max_try_cnt = 10
        timeout = 25
        
        while try_cnt < max_try_cnt:
            try:
                time.sleep(3)
                response = openai.ChatCompletion.create(
                                    model=self.model_name,
                                    messages=message,
                                    **self.generation_params)
                text_response = response["choices"][0]["message"]["content"]
                return text_response
            except Exception as e:
                wait_time = (estimated_tokens / rate_limit) * 60 * (1 + try_cnt**2 / 4)
                print(f"Error {str(e)} occurred. Waiting for {wait_time} seconds...")
                time.sleep(wait_time)
                try_cnt += 1
        if try_cnt ==10:
            error_message = "Errors occurred too many times. Aborting..."
            print(error_message)
            return(error_message)
        
    
class DebugModel(LanguageModel):
    def __init__(self):
        pass

    def __repr__(self):
        return f"DebugModel(model_name={self.model_name}, generation_params={self.generation_params})"
    
    def query(self, prompt : str) -> str:
        print(f"Model Recieved: {prompt}")
    

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
    COMMON_SENSE = auto()
    COMMON_SENSE_CoT = auto()
    GSM8K = auto()
    MBPP = auto()
    RTE = auto()
    MNLI = auto()

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
class BatchPromptingDebugConfig:
    truncate_examples : bool = False,
    truncate_batch_queries : bool = False
    save_batched_model_inputs : Optional[Path] = None
    save_batched_model_outputs : Optional[Path] = None
@dataclass
class BatchPromptingExperimentConfig:
    # can either load a dataset from huggingface or from a local json file
    questions_dataset_config : DatasetConfig
    examples_dataset_config : DatasetConfig
    # can also choose to load a dataset from a file for questions or examples
    task_description: str
    # we specify baseline as a boolean as we might have a batch of size 1 at the end, but it will still
    # use the batched prompt template rather than the baseline prompt template
    k_shot: int
    batch_size: int
    example_selection: ExampleSelectionType 
    question_format: EXAMPLE_FORMAT_FUNCTION_TYPE
    example_question_format: EXAMPLE_FORMAT_FUNCTION_TYPE
    example_answer_format: EXAMPLE_FORMAT_FUNCTION_TYPE
    # Note that this will include the api model name whereas model_name will be less formal like GPT-3.5 vs LLama-2-70B, etc.
    model_api: ModelAPIType
    generation_params: GENERATION_PARAMETERS_TYPE
    debug : Optional[BatchPromptingDebugConfig] = None
    pre_question_instructions: Optional[str] = None
    prompt_format: Optional[Callable[[Dict[str, str]], str]] = None
    is_baseline: bool = False
    random_seed: int = 0


class BatchPromptExperiment:
    def __init__(
            self,
            config: BatchPromptingExperimentConfig,
    ):
        self.config = config
        self.questions = self.load_dataset(self.config.questions_dataset_config)
        self.examples = self.load_dataset(self.config.examples_dataset_config)
        # self.examples = load_dataset(
        #     *self.config.hf_dataset_path,
        #     split=self.config.examples_split_name,
        # )
        # self.questions = load_dataset(
        #     *self.config.hf_dataset_path,
        #     split=self.config.questions_split_name,
        # )
        # must add an index column to gsm8k
        self.debug = self.config.debug
        self.batch_prompt_template = BatchPromptTemplate(
            examples=self.examples,
            dataset=self.config.questions_dataset_config.dataset,
            task_description=self.config.task_description,
            pre_question_instructions=self.config.pre_question_instructions,
            num_questions=self.config.batch_size,
            num_examples=self.config.k_shot,
            question_format=self.config.question_format,
            example_question_format=self.config.example_question_format,
            example_answer_format=self.config.example_answer_format,
            prompt_format=self.config.prompt_format,
            example_selection=self.config.example_selection,
            debug=self.debug,
            is_baseline=self.config.is_baseline,
        )
        self.model = self.load_language_model(
            model_api=self.config.model_api, 
            generation_params=self.config.generation_params
        )
        random.seed(self.config.random_seed)

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
                warnings.filterwarnings('ignore', category=PicklingWarning)

                dataset = dataset.filter(lambda example: example['task'] == dataset_config.task_name_for_CoT_filter)
                if dataset_config.task_name_for_CoT_filter =="rte":
                    dataset = dataset.filter(lambda example: "Question with options: can we draw the following hypothesis from the context?" in example['source'])
                if dataset_config.task_name_for_CoT_filter =="mnli":
                    dataset = dataset.filter(lambda example: "Premise: " in example['source'] and "Hypothesis: " in example['source'])
        # add an index column to gsm8k
        if dataset_config.dataset == DatasetType.GSM8K:
            dataset = dataset.add_column('idx', list(range(len(dataset))))
        return dataset
    
    def load_language_model(self, model_api, generation_params) -> LanguageModel:
        match model_api:
            case ModelAPIType.OPEN_AI:
                token = read_api_token(Path("data/imported/open_ai_token.txt"))
                model = OpenAIModel(
                    api_token=token,
                    model_name=generation_params['model_name'],
                    generation_params=generation_params
                )
            case ModelAPIType.TOGETHER_AI:
                # together.ai
                token = read_api_token(Path("data/imported/together_ai_token.txt"))
                model = TogetherAIModel(
                    api_token=token,
                    model_name=generation_params['model_name'],
                    generation_params=generation_params
                )
            case ModelAPIType.DEBUG:
                model = DebugModel()
            # otherwise
            case _: 
                raise NotImplementedError("Only OpenAI and TogetherAI APIs are currently supported")
        # cover all bases
        return model

    
    def batch_query_model(
        self, 
        model_inputs: List[Tuple[List[int], str]]
    ) -> List[Tuple[List[int], str]]:
        
        if self.model is None:
            self.model = self.load_language_model()

        messages = [[{"role": "user", "content": i[1]}] for i in model_inputs]
        attempt_cnt = 0
        max_attempts = 10 
        if "llama" in self.model.model_name:
            model_query_batch_size = 2
        else:
            model_query_batch_size = 10
        results = []
        message_sublists = [messages[i:i+model_query_batch_size] for i in range(0, len(messages), model_query_batch_size)]
        for i, batched_messages in enumerate(message_sublists):
            if "llama" in self.model.model_name:
                time.sleep(2.2)
            while attempt_cnt < max_attempts:
                try:
                    responses = batch_completion(
                    model=self.model.model_name,
                    messages = batched_messages,
                    **self.model.generation_params,
                    # temperature=0,
                    # max_tokens=None,
                    )
                    curr_results = [response["choices"][0]["message"]["content"] for response in responses]
                    results.extend(curr_results)
                    break
                except Exception as e:
                    attempt_cnt += 1
                    print(f"Error {str(e)} occurred. Retrying...")
                    time.sleep(30* attempt_cnt)
            if attempt_cnt == max_attempts:
                curr_results = ["" for i in range(len(batched_messages))]
                results.extend(curr_results)
        return results


    def batch_questions(self) -> List[Tuple[List[ID_TYPE], List[Dict[str, Any]]]]:

        batched_dataset : List[Dict[str, List[Any]]] = [   
            self.questions[i:i+self.config.batch_size]
            for i in range(0, len(self.questions), self.config.batch_size)
        ]
        if "CommonsenseQA" in self.config.task_description:
            idx = 0
            for batch in batched_dataset:
                batch['idx'] = [idx + i for i in range(len(batch['question']))]
                idx += len(batch['question'])
        batched_questions : List[Tuple[List[ID_TYPE], List[Dict[str, Any]]]] = []
        for batch in batched_dataset:
            ids = batch[DATASET_ID_KEYS[self.config.questions_dataset_config.dataset][0]]
            questions = [
                {key : batch[key][i] for key in batch.keys()}
                for i in range(len(ids))
            ]
            batched_questions.append((ids, questions))

        return batched_questions
    
    def answers_from_batched_questions(
        self, 
        batched_questions: List[Tuple[List[ID_TYPE], List[Dict[str, Any]]]]
    ) -> Dict[ID_TYPE, Dict[str, Any]]:
        answers :  Dict[ID_TYPE, Dict[str, Any]] = {
            question_id : question
            for (ids, questions) in batched_questions
            for (question_id, question) in zip(ids, questions)
        }
        return answers


    def execute(self) -> Tuple[List[Tuple[List[ID_TYPE], str]], Dict[ID_TYPE, Dict[str, Any]]]:
        """
        X Load Dataset
        X Generate set of model inputs (using dataset + config + FlexiblePromptTemplate)
        X query model for each input (save raw outputs to a file somewhere)
        """
        # splits self.questions into batches that are lists of individual dictionaries along with their ids
        batched_questions: List[Tuple[List[ID_TYPE], List[Dict[str, Any]]]] = self.batch_questions()
        answers_dict = self.answers_from_batched_questions(batched_questions)

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
        return (batched_model_inputs, batched_model_outputs, answers_dict)


def parse_answers(model_outputs: List[Tuple[List[ID_TYPE], str]]) -> Dict[List[ID_TYPE], str]:
    raise NotImplementedError()

class BatchPromptTemplate:
    # write the docs for this
    """
    BatchPromptTemplate is a class that generates prompts for a batch of examples.
    It is used to generate prompts for the k-shot experiment.
    Args:
    - Examples: a huggingface dataset of examples
    - task_description: a string describing the task - this goes at the top level of the prompt
    - num_questions: the number of questions to ask per prompt
    - num_examples: the number of examples to include in each prompt, > 1 means batched
    - example_template: a PromptTemplate that takes in a dataset's features dictionary and returns a string,
        it can also optionally take in an index [i] for batched prompts. This function will be used 
        both for building/retrieving from the example database and for generating prompts.
    - example_selection: an enum that specifies how to select examples for the prompt

    """
    def __init__(
            self,
            examples: Dataset,
            dataset: DatasetType,
            task_description: str,
            pre_question_instructions: str,
            num_questions: int,
            num_examples: int,
            is_baseline: bool,
            # the optional int is the index of the example in the batch, none if baseline
            question_format: Callable[[Dict[str, Any], Optional[int]], str],
            example_question_format: Callable[[Dict[str, Any], Optional[int]], str],
            example_answer_format: Callable[[Dict[str, Any], Optional[int]], str],
            example_selection: ExampleSelectionType,
            debug: Optional[BatchPromptingDebugConfig] = None,
            prompt_format: Optional[Callable[[Dict[str, str]], str]] = None,
    ):
        self.examples = examples
        self.dataset = dataset
        self.task_description = task_description
        self.pre_question_instructions = pre_question_instructions
        self.num_questions = num_questions
        self.num_examples = num_examples
        self.question_format = question_format
        self.example_question_format = example_question_format
        self.example_answer_format = example_answer_format
        self.prompt_format = prompt_format
        self.example_selection = example_selection
        self.is_baseline = is_baseline

        if self.is_baseline:
            assert(self.num_questions == 1)

        match self.example_selection:
            case ExampleSelectionType.RANDOM:
                pass
            case ExampleSelectionType.LEXICAL:
                raise NotImplementedError("Lexical example selection is not yet implemented")
            case ExampleSelectionType.SEMANTIC | ExampleSelectionType.MAX_MARGINAL_RELEVANCE:
                selector_class = {
                    ExampleSelectionType.SEMANTIC: SemanticSimilarityExampleSelector,
                    ExampleSelectionType.MAX_MARGINAL_RELEVANCE: MaxMarginalRelevanceExampleSelector,
                }[self.example_selection]
                print("Initializing Semantic Example Selector...")

                examples = list(self.examples)
                if debug:
                    if debug.truncate_batch_queries:
                        examples = examples[:150]

                self.example_selector = selector_class.from_examples(
                    # Need to turn hf dataset into a list of dicts
                    examples=examples,
                    embeddings=HuggingFaceEmbeddings(),
                    vectorstore_cls=Chroma,
                    k=self.num_examples,
                    input_keys=DATASET_INPUT_KEYS[self.dataset],
                )
                print("Done initializing Semantic Example Selector...")

    def get_examples(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # batch is a list of dataset instances that will have the same keys as the dataset
        match self.example_selection:
            case ExampleSelectionType.RANDOM:
                return [self.examples[i] for i in random.sample(range(len(self.examples)), self.num_examples)] 
            case ExampleSelectionType.LEXICAL:
                raise NotImplementedError("Lexical example selection is not yet implemented")
            case ExampleSelectionType.SEMANTIC | ExampleSelectionType.MAX_MARGINAL_RELEVANCE:
                top_k_examples_per_question = [
                    self.example_selector.select_examples(question)
                    for question in batch
                ]
                # num questions
                # note that we don't use self.num_questions because the batch could 
                # be smaller than that if it's the last batch
                b = len(top_k_examples_per_question)
                # take the first k examples, one from each question looping if necessary
                batch_examples = [
                    top_k_examples_per_question[i % b][i // b]
                    for i in range(self.num_examples)
                ]
                assert self.num_examples == len(batch_examples)
                return batch_examples

    def generate_prompt(self, batch: List[Dict[str, Any]]) -> str:
        example_questions = []
        example_answers = []
        questions = []

        examples = self.get_examples(batch)
        for i, example in enumerate(examples):
            # the format functions take care of the Q[i] notation
            example_questions.append(self.example_question_format(example, i))
            example_answers.append(self.example_answer_format(example, i))
        
        for i, question in enumerate(batch):
            questions.append(self.question_format(question, i))
        

        if self.prompt_format is not None:
            fields = {
                'task_description' : self.task_description,
                'example_questions' : example_questions,
                'example_answers' : example_answers,
                'pre_questions_instructions' : self.pre_question_instructions,
                'questions' : questions,
            }
            prompt = self.prompt_format(fields)
        else:
            if self.is_baseline:
                examples = [
                    item 
                    for pair in zip(example_questions, example_answers) 
                    for item in pair
                ]
            else:
                examples = [
                    *example_questions,
                    *example_answers,
                ]
            if self.num_examples ==0:
                prompt = (
                    self.task_description.format(batch_size = self.num_questions, few_shot_examples = "") + '\n'.join(questions))
            else:
                few_shot_example_prompt_part = self.pre_question_instructions+"\n".join(examples) + "\n"
                extra_description_to_distinguish_examples_from_actual = "(Note: This is the actual task. Use the format shown in the few-shot examples to provide your answers for the following questions.)\n"
                prompt = (
                    self.task_description.format(batch_size = self.num_questions, few_shot_examples = few_shot_example_prompt_part) + extra_description_to_distinguish_examples_from_actual + '\n'.join(questions))
        return prompt


from src.utils.parsing_functions import * 
questions_config_rte = DatasetConfig(
    dataset=DatasetType.RTE,
    hf_dataset_path=['glue', 'rte'],
    split_name='validation',
)
examples_config_rte = DatasetConfig(
    dataset=DatasetType.RTE,\
    hf_dataset_path=['kaist-ai/CoT-Collection'],
    task_name_for_CoT_filter = "rte",
    # hf_dataset_path=['glue', 'rte'],
    split_name='train',
)
questions_config_GSM8K = DatasetConfig(
    dataset=DatasetType.GSM8K,
    hf_dataset_path=['gsm8k', 'main'],
    split_name='test',
)
examples_config_GSM8K = DatasetConfig(
    dataset=DatasetType.GSM8K,
    hf_dataset_path=['gsm8k', 'main'],
    split_name='train',
)
questions_config_GSM8K_HARD = DatasetConfig(
    dataset=DatasetType.GSM8K_HARD,
    hf_dataset_path=["reasoning-machines/gsm-hard"],
    split_name='train',
)
examples_config_GSM8K_HARD = DatasetConfig(
    dataset=DatasetType.GSM8K_HARD,
    hf_dataset_path=["reasoning-machines/gsm-hard"],
    split_name='train',
)
task_description_GSM8K_HARD = '''Solve the following math question. # Instruction: For each question in the batch, provide a single answer, following the format A[index]: answer. Output only the answers with the associated index in A[idx]: answer format.'''
questions_config_MBPP = DatasetConfig(
    dataset=DatasetType.MBPP,
    hf_dataset_path=['mbpp'],
    split_name='validation',
)
examples_config_MBPP = DatasetConfig(
    dataset=DatasetType.MBPP,
    hf_dataset_path=['mbpp'],
    split_name='train',
)
task_description_MBPP = '''You are tasked with solving Python programming problems that are designed to be solvable by entry-level programmers. Each problem will consist of a task description, and your job is to output a string that when parsed is an executable Python code function that fulfills the requirements of the task. # Instruction: For each question in the batch, provide a single answer, following the format A[index]: answer. Output only the answers with the associated index in "A[idx]: answer" format.'''

questions_config_MNLI = DatasetConfig(
    dataset=DatasetType.MNLI,
    hf_dataset_path=['glue', 'mnli'],
    split_name='validation_matched',
)

examples_config_MNLI = DatasetConfig(
    dataset=DatasetType.MNLI,\
    hf_dataset_path=['kaist-ai/CoT-Collection'],
    task_name_for_CoT_filter = "mnli",
    # hf_dataset_path=['glue', 'rte'],
    split_name='train',
)

questions_config_COMMON_SENSE = DatasetConfig(
    dataset=DatasetType.COMMON_SENSE,
    hf_dataset_path=['commonsense_qa'],
    split_name='validation',
)
examples_config_COMMON_SENSE = DatasetConfig(
    dataset=DatasetType.COMMON_SENSE,
    hf_dataset_path=['kaist-ai/CoT-Collection'],
     task_name_for_CoT_filter = "commonsenseqa",
    split_name='train',
)

config_to_answer_type = {"GSM8K": "numerical", 
                "GSM8K_HARD": "numerical", 
                "COMMON_SENSE": "categorical", 
                "MBPP": "code",
                "MNLI": "binary",
                "RTE": "categorical"}
config_param_list = { 
    "rte": [questions_config_rte, examples_config_rte, task_description_rte, rte_question_format, rte_answer_format, rte_example_question_format],
    "GSM8K": [questions_config_GSM8K, examples_config_GSM8K, task_description_GSM8K, gsm8k_question_format, gsm8k_answer_format, gsm8k_example_question_format],
    # # "MBPP": [questions_config_MBPP, examples_config_MBPP, task_description_MBPP, mbpp_question_format, mbpp_answer_format],
    "MNLI": [questions_config_MNLI, examples_config_MNLI, task_description_MNLI, mnli_question_format, mnli_answer_format, mnli_example_question_format],
    # #"GSM8K_HARD": [questions_config_GSM8K_HARD, examples_config_GSM8K_HARD, task_description_GSM8K_HARD, gsm8k_question_format, gsm8k_answer_format],
    "COMMON_SENSE": [questions_config_COMMON_SENSE, examples_config_COMMON_SENSE, task_description_COMMON_SENSE, commonsense_question_format, commonsense_answer_format, commonsense_example_question_format],
}

def extract_math_answers(text):
    # Initialize an empty list to store the extracted numbers along with their positions
    extracted_numbers = []
    
    # Regex pattern with a fallback capturing group (.+?) to capture any string that appears after "The answer is"
    # pattern = r"The answer is\s*(-?\$?|\$?-?)([\d,.-]+)|The answer is\s*(.+?)\."
    
    pattern = r'The answer is\s*(-?)\$?(\d{1,3}(?:,\d{3})*)'

    # Find all matches of the pattern in the text using finditer
    matches = re.finditer(pattern, text)
    
    # Extract numbers and their positions from the matches
    for match in matches:
        position = match.start()
        
        # Check if the match is a parsable number
        if match.group(2):
            try:
                # Determine if the number is negative based on the presence of the negative sign in the first capturing group
                is_negative = '-' in match.group(1)
                
                # Remove commas and convert to integer
                num = int(match.group(2).replace(',', ''))
                
                # Apply the negative sign if needed
                if is_negative:
                    num = -num
                
                extracted_numbers.append((position, num))
            except ValueError:
                extracted_numbers.append((position, None))
        else:
            # If the match is not a parsable number, insert None
            extracted_numbers.append((position, None))
    
    # Sort the extracted numbers based on their positions in the text
    extracted_numbers.sort(key=lambda x: x[0])
    
    # Return only the numbers or None placeholders, in the order they appeared in the text
    return [num for position, num in extracted_numbers]


def extract_answers_batch(output_str: str, answer_type = None) -> List[int]:
    answer_type = answer_type.upper()
    if answer_type == "COMMON_SENSE":
                # Initialize an empty list to store the extracted answers.
        answers = []
        
        # Step 1: Split the string by newlines to process each line individually.
        lines = output_str.strip().split("\n")
        
        # Modified regex pattern to extract either an integer or single uppercase letter
        # after ": ".
        primary_pattern = r": ([A-D]|\d+)"
        
        # Backup regex pattern to extract any number in the line.
        backup_pattern = r"([A-D]|\d+)"
        
        if answer_type == "COMMON_SENSE":
            raise None  # Raise appropriate Exception or modify as needed

        # Step 2: Loop through each line to extract the answer.
        for line in lines:
            # Try primary regex pattern first.
            match = re.search(primary_pattern, line)
            if match:
                answers.append(match.group(1))
            else:
                # If primary fails, try the backup pattern.
                match = re.search(backup_pattern, line)
                if match:
                    answers.append(match.group(1))

        return answers
    elif answer_type ==  "MBPP":
        raise None
    elif answer_type in ["GSM8K_HARD", "GSM8K"]:
        answers = extract_math_answers(output_str)
        return answers
    elif answer_type in ["RTE", "MNLI"]:
        answers = []
        
        # Split the string by newlines to process each line individually.
        lines = output_str.strip().split("\n")
        
        # Loop through each line to extract the answer.
        for line in lines:
            if line =='':
                continue
            answer = extract_last_number(line)
            if answer == None and "Q[" in line:
                continue
            answers.append(answer)
        return answers

    else: 
        raise NotImplementedError()

def convert_to_int(input_str: str) -> int:
    # Step 1: Remove unnecessary characters like spaces and commas
    cleaned_str = re.sub(r"[^\d.-]", "", input_str)
    
    # Step 2: Convert the cleaned string to float
    float_val = float(cleaned_str)
    
    # Step 3: Convert the float to integer
    int_val = int(float_val)
    
    return int_val

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
 
def get_index_to_ground_truth(answers_dict, task_name):
    task_name = task_name.upper()
    index_to_ground_truth = {}
    for answer_index, answer in answers_dict.items():
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
def get_index_to_pred(batched_model_inputs, batched_model_outputs, task_name):
    index_to_pred = {}
    for batch_input, batch in zip(batched_model_inputs, batched_model_outputs):
        indices = batch_input[0]
        LLM_output = batch
        
        text_split_by_batch = split_answers(LLM_output)
        if task_name == "COMMON_SENSE":
            answers = [extract_last_letter(text) for text in text_split_by_batch]
        else: 
            answers = [extract_last_integer(text) for text in text_split_by_batch]
        # answers = extract_answers_batch(LLM_output, task_name)
        # answers = parse_batched_answers(LLM_output, task_name)
        if len(answers) == len(indices):
            for index, answer in zip(indices, answers):
                index_to_pred[index] = answer
        elif len(answers) > len(indices):
            for index, answer in zip(indices, answers[0:len(indices)]):
                index_to_pred[index] = answer 
        else:
            for index, answer in zip(indices, answers[0:len(indices)]):
                index_to_pred[index] = answer
            
            for index in indices[len(answers):]:
                index_to_pred[index] = None
 
    return index_to_pred
from typing import Optional

def extract_last_number(text: str) -> Optional[int]:
    # Define the regex pattern to capture numbers with optional negative sign, dollar sign, and commas
    pattern = r"[-$]?[\d,]+"

    # Find all matching numbers in the string
    matches = re.findall(pattern, text)

    # If no match is found, return None
    if not matches:
        return None
    try:
        # Grab the last match
        last_match = matches[-1]

        # Remove dollar sign and commas, if any
        cleaned_match = last_match.replace("$", "").replace(",", "")

        # Convert to int
        final_number = int(cleaned_match)
        return final_number
    except Exception as e:
        return None


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



def run_batched_tests(config, config_to_answer_type):
    task_to_stats ={}
    batch_sizes = [1, 4, 8]
    k_shot_sizes = [0,1, 6]
# 'gpt-4-1106-preview': 
# {"param_object": OpenAIGenerationParameters(
#     # model_name='gpt-3.5-turbo',
#     # model_name='gpt-3.5-turbo-16k',
#     model_name='gpt-4-1106-preview',
#     temperature=0,
#     max_tokens=None,
#     frequency_penalty=1.0,),
# "model_api": ModelAPIType.OPEN_AI},

    model_params = {
                    "LLAMA-2-13B":  {"param_object": TogetherAIGenerationParameters(
                    model_name='together_ai/togethercomputer/llama-2-13b-chat',
                    temperature = 0,
                    # temperature=0.6,
                    # max_tokens=64,
                    max_tokens=1600, # 4096 max but some tokens must be used for input
                    frequency_penalty=1.0),
                    "model_api": ModelAPIType.TOGETHER_AI},
                    "gpt-3.5-turbo-16k": 
                    {"param_object": 
                        OpenAIGenerationParameters(
                        # model_name='gpt-3.5-turbo',
                        model_name='gpt-3.5-turbo-16k',
                        # model_name='gpt-4',
                        # temperature=0,
                        temperature=.3,
                        max_tokens=None,
                        frequency_penalty=1.0,), 
                    "model_api": ModelAPIType.OPEN_AI}, 
                    'gpt-4': 
                    {"param_object": OpenAIGenerationParameters(
                        # model_name='gpt-3.5-turbo',
                        # model_name='gpt-3.5-turbo-16k',
                        model_name='gpt-4',
                        temperature=0,
                        max_tokens=None,
                        frequency_penalty=1.0,),
                    "model_api": ModelAPIType.OPEN_AI},
                    "LLAMA-2-70B": {
                        "param_object": TogetherAIGenerationParameters(
                        model_name='together_ai/togethercomputer/llama-2-70b-chat',
                        temperature = 0,
                        # temperature=0.6,
                        # max_tokens=64,
                        max_tokens=2000,
                        frequency_penalty=1.0,
                        ),
                        "model_api": ModelAPIType.TOGETHER_AI}
                    }

    # Note: This overestimates if DEBUG_NUM_QUESTIONS_WANT_ANSWER_PER_EXPERIMENT> dataset size
    total_num_batched_complete_api_calls = len(model_params.keys()) * len(k_shot_sizes)* sum( [-(-max(1,.1*DEBUG_NUM_QUESTIONS_WANT_ANSWER_PER_EXPERIMENT)//batch_size) for batch_size in batch_sizes])
    estimated_batch_response_time = 60 # in seconds
    expected_execution_time_minutes = total_num_batched_complete_api_calls * estimated_batch_response_time / 60
    GPT_COST_USD_PER_QUERY = .01
    NUM_CALLS_IN_BATCH = 10
    expected_cost_USD = 2 * GPT_COST_USD_PER_QUERY * NUM_CALLS_IN_BATCH * total_num_batched_complete_api_calls
    start_time = time.time()
    print(f"Number of BatchCompleteAPI Calls: {total_num_batched_complete_api_calls}")
    print(f"Expected execution time in minutes: {expected_execution_time_minutes}" )
    print(f"Expected API Call Cost: {expected_cost_USD}" )

    os.environ['TOGETHERAI_API_KEY'] = read_api_token(Path("data/imported/together_ai_token.txt"))
    os.environ['OPENAI_API_KEY'] = read_api_token(Path("data/imported/open_ai_token.txt"))
    for task_name, configs in config_param_list.items():
        for k_shot_size in k_shot_sizes:
            for batch_size in batch_sizes:
                for model_name, model_param in model_params.items():
                    # These two checks are to ensure that we don't exceed the context window of the models (and also to reduce API calls)
                    if model_name == "LLAMA-2-70B" and k_shot_size + batch_size > 5:
                        continue
                    elif model_name == "gpt-4" and k_shot_size + batch_size > 10:
                        continue
                    elif k_shot_size + batch_size > 20:
                        continue
                    
                    questions_config, examples_config, task_description, question_format, answer_format , example_question_format = configs

                    config = BatchPromptingExperimentConfig(
                    questions_dataset_config=questions_config,
                    examples_dataset_config=examples_config,
                    task_description=task_description,
                    pre_question_instructions="Consider the following examples and maintain their formatting.#### Few-Shot Examples for Demonstration:\n(Note: These examples are for illustration only and are not to be answered as part of the task.)\n",
                    k_shot=k_shot_size,
                    example_selection=ExampleSelectionType.RANDOM,
                    # example_selection=None,
                    question_format = question_format,
                    example_question_format=example_question_format,
                    example_answer_format=answer_format,
                    batch_size=batch_size,
                    model_api=model_param["model_api"],
                    # model_api=ModelAPIType.TOGETHER_AI,
                    generation_params=model_param["param_object"],

                    # generation_params=llama_2_70B_gen_params,
                    random_seed=0,
                    debug=BatchPromptingDebugConfig(
                            truncate_examples=True,
                            truncate_batch_queries=True,
                            save_batched_model_inputs=None,
                            save_batched_model_outputs=None,
                        )
                    )
                    experiment = BatchPromptExperiment(config)
                    batched_model_inputs, batched_model_outputs, answers_dict = experiment.execute()
                    if task_name == "MBPP":
                        evaluator = CodeEvaluator()
                        raise NotImplementedError()
                        # mbpp_code_example = mbpp['train']['code'][index]
                        # mbpp_test_cases_example = mbpp['train']['test_list'][index]
                        # result = evaluator.run_code_and_tests(mbpp_code_example, mbpp_test_cases_example)
                    else:
                        pred, ground_truth = get_pred_ground_truth(batched_model_inputs, batched_model_outputs, answers_dict, task_name)
                        stat = get_stats(y_pred=pred, y_true=ground_truth, answer_type = config_to_answer_type[task_name.upper()])

                    if task_name not in task_to_stats:
                        task_to_stats[task_name] = {}
                    if k_shot_size not in task_to_stats[task_name]:
                        task_to_stats[task_name][k_shot_size] = {}
                    if batch_size not in task_to_stats[task_name][k_shot_size]:
                        task_to_stats[task_name][k_shot_size][batch_size] = {}
                    if model_name not in task_to_stats[task_name][k_shot_size][batch_size]:
                        task_to_stats[task_name][k_shot_size][batch_size][model_name] = {}
                    
                    task_to_stats[task_name][k_shot_size][batch_size][model_name]["batched_model_inputs"] = batched_model_inputs
                    task_to_stats[task_name][k_shot_size][batch_size][model_name]["batched_model_outputs"] = batched_model_outputs
                    task_to_stats[task_name][k_shot_size][batch_size][model_name]["stat"] = stat
                    task_to_stats[task_name][k_shot_size][batch_size][model_name]["pred"] = pred
                    task_to_stats[task_name][k_shot_size][batch_size][model_name]["ground_truth"] = ground_truth
            print(f"End of experiment for k_shot_size {k_shot_size}")
    end_time = time.time()
    print(f"Expected total time taken in minutes: {round(expected_execution_time_minutes)}", )
    print(f"Actual total time taken in minutes: {round((end_time - start_time)/60)}")
    with open("task_to_stats_all_tasks_all_models", 'wb') as f:
        pickle.dump(task_to_stats, f)
    with open("task_to_stats_all_tasks_all_models_backup", 'wb') as f:
        pickle.dump(task_to_stats, f)
    print(task_to_stats)
    return task_to_stats

if __name__ == "__main__":
    run_batched_tests(config_param_list, config_to_answer_type)