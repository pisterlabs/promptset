import random
import pickle
from src.models.model_api import (
    read_api_token, 
    LanguageModel, 
    TogetherAIModel, 
    OpenAIModel, 
    DebugModel,
    TogetherAIGenerationParameters, 
    OpenAIGenerationParameters, 
    DebugGenerationParameters,
)
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
#the keys over which examples are semantically retrieved. 
# NOTE: these keys must be string/stringable (not dicts like some of commonsenseqa's)
DATASET_EXAMPLE_KEYS = {
    DatasetType.GSM8K_HARD_CoT : ['question'],
    DatasetType.GSM8K_HARD : ['input'],
    DatasetType.COMMON_SENSE : ['question'],
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
    truncate_batch_queries : int = False
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

# datasets = ['GSM8K', 'MBPP', 'glue-RTE', 'glue-MNLI']

class BatchPromptExperiment:
    def __init__(
            self,
            config: BatchPromptingExperimentConfig,
    ):
        self.config = config
        self.questions = self.load_dataset(self.config.questions_dataset_config)
        self.examples = self.load_dataset(self.config.examples_dataset_config)
        self.debug = self.config.debug
        self.batch_prompt_template = BatchPromptTemplate(
            examples=self.examples,
            dataset=self.config.questions_dataset_config.dataset,
            task_description=self.config.task_description,
            pre_question_instructions=self.config.pre_question_instructions,
            num_questions=self.config.batch_size,
            num_examples=self.config.k_shot,
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
        # add an index column to the dataset if it doesn't already have one
        id_key_name = DATASET_ID_KEYS[dataset_config.dataset][0]
        if id_key_name not in dataset.column_names:
            dataset = dataset.add_column(id_key_name, list(range(len(dataset))))
        
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
        model_inputs: List[Tuple[List[ID_TYPE], str]], 
    ) -> List[Tuple[List[ID_TYPE], str]]:

        # create the model API object if it doesn't exist
        if self.model is None:
            self.model = self.load_language_model()

        batched_results = [
            (ids, self.model.query(prompt))
            for (ids, prompt) in tqdm(model_inputs)
        ]
        return batched_results

    def batch_questions(self) -> List[Tuple[List[ID_TYPE], List[Dict[str, Any]]]]:

        batched_dataset : List[Dict[str, List[Any]]] = [   
            self.questions[i:i+self.config.batch_size]
            for i in range(0, len(self.questions), self.config.batch_size)
        ]
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
            batched_model_inputs = batched_model_inputs[:3]
        batched_model_outputs = self.batch_query_model(batched_model_inputs)
        if batched_model_inputs:
            if self.debug.save_batched_model_inputs:
                pickle.dump((batched_model_inputs), open(self.debug.save_batched_model_inputs, 'wb'))
            if self.debug.save_batched_model_outputs:
                pickle.dump((batched_model_outputs), open(self.debug.save_batched_model_outputs, 'wb'))


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
        self.example_question_format = example_question_format
        self.example_answer_format = example_answer_format
        self.prompt_format = prompt_format
        self.example_selection = example_selection
        self.is_baseline = is_baseline
        self.debug = debug

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
                        examples = examples[:20]

                # this ensures that only the keys that are used for example selection (which must be strings/stringable are loaded
                # and add a secret index column to later get the full example from the dataset
                selection_examples = self.examples.select_columns(
                    DATASET_EXAMPLE_KEYS[self.dataset]
                )
                selection_examples = selection_examples.add_column(
                    "_idx", list(range(len(selection_examples)))
                )
                
                self.example_selector = selector_class.from_examples(
                    # Need to turn hf dataset into a list of dicts
                    examples=selection_examples,
                    embeddings=HuggingFaceEmbeddings(),
                    vectorstore_cls=Chroma,
                    k=self.num_examples,
                    # DATASET_EXAMPLE_KEYS may be different from input keys
                    # because they can only be strings, so commonsense, for example, has a dictionary of choices that we won't use
                    input_keys=DATASET_EXAMPLE_KEYS[self.dataset],
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
                # get the top k examples for each question
                # by first retrieving from the selection_examples for each question which have the associated index
                # then using that index to get the real examples
                top_k_examples_per_question : List[List[Dict[str, Any]]] = [
                    [
                        self.examples[retrieved['_idx']]
                        for retrieved in self.example_selector.select_examples(question)
                    ]
                    for question in batch
                ]
                if self.debug:
                    print("Top k examples per question:")
                    print(top_k_examples_per_question)
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
            questions.append(self.example_question_format(question, i))

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
            # will be empty and provide nothing to the prompt if pre_questions_instructions is None
            pre_questions_instructions = [self.pre_question_instructions] if self.pre_question_instructions is not None else []
            prompt = "\n".join(
                [
                    self.task_description,
                    *examples,
                    *pre_questions_instructions,
                    *questions,
                ]
            )
        return prompt



if __name__ == "__main__":
    
    # from data.parsing_functions import *
    example_question_format = lambda example, i: f"Premise[{i}]: {example['sentence1']}\nHypothesis[{i}]: {example['sentence2']}"
    example_answer_format = lambda example, i: f"Answer[{i}]: {example['label']}"
    example_question_format_baseline = lambda example, i: f"Premise: {example['sentence1']}\nHypothesis: {example['sentence2']}"
    example_answer_format_baseline = lambda example, i: f"Answer: {example['label']}"

    oai_gen_params = OpenAIGenerationParameters(
            model_name='gpt-3.5-turbo',
            temperature=0.6,
            max_tokens=64,
            frequency_penalty=1.0,
        )

    questions_config_rte = DatasetConfig(
        dataset=DatasetType.RTE,
        hf_dataset_path=['glue', 'rte'],
        split_name='validation',
    )
    examples_config_rte = DatasetConfig(
        dataset=DatasetType.RTE,
        hf_dataset_path=['glue', 'rte'],
        split_name='train',
    )
    task_description_rte = 'Determine whether the hypothesis is entailed by the premise. Answer 0 for entailed, and 1 for not entailed.'

    questions_config_COMMON_SENSE = DatasetConfig(
        dataset=DatasetType.COMMON_SENSE,
        hf_dataset_path=['commonsense_qa'],
        split_name='validation',
    )
    examples_config_COMMON_SENSE = DatasetConfig(
        dataset=DatasetType.COMMON_SENSE,
        hf_dataset_path=['commonsense_qa'],
        split_name='train',
    )
    task_description_COMMON_SENSE = '''You are tasked with answering multiple-choice questions that require both contextual understanding and general world knowledge. Each question will have five options labeled 'a', 'b', 'c', 'd', and 'e'. Your job is to select the most appropriate answer by outputting the letter corresponding to that option. " These questions are part of the CommonsenseQA dataset, designed to test your ability to answer questions that often require prior knowledge. Instruction: For each question in the batch, provide a single answer, following the format A[index]: answer. Output only the answers with the associated index in "A[idx]: answer" format. '''

    def commonsense_question_format(example, i):
        question_with_answer_choices = example["source"]
        rationale = example["choices"]
        def build_question_string(question, choices, i):
            question_str = f"Question[{i}]: {question}\nAnswer Choices: \n"
            choice_labels = choices['label']
            choice_texts = choices['text']
            
            for label, text in zip(choice_labels, choice_texts):
                question_str += f"{label}: {text}\n"
            
            return question_str
        question_str = build_question_string(question_with_answer_choices, rationale)
        return question_str

    def commonsense_answer_format(example, i):
        answer = example["answerKey"]
        return f"Answer[{i}]: {answer}"

    config = BatchPromptingExperimentConfig(
        questions_dataset_config=questions_config_COMMON_SENSE,
        examples_dataset_config=examples_config_COMMON_SENSE,
        task_description=task_description_COMMON_SENSE,
        pre_question_instructions="Consider the following examples and maintain their formatting.\n",
        k_shot=7,
        example_selection=ExampleSelectionType.SEMANTIC,
        example_question_format=commonsense_question_format,
        example_answer_format=commonsense_answer_format,
        batch_size=4,
        model_api=ModelAPIType.OPEN_AI,
        generation_params=oai_gen_params,
        random_seed=0,
        is_baseline=False,
        debug=BatchPromptingDebugConfig(
            truncate_examples=True,
            truncate_batch_queries=True,
            save_batched_model_inputs=Path('batched_model_inputs.pkl'),
            save_batched_model_outputs=Path('batched_model_outputs.pkl'),
        ),
    )
    experiment = BatchPromptExperiment(config)
    experiment.execute()