import openai
from typing import List, Tuple, Union
from abc import ABC, abstractmethod
from pathlib import Path
from utils.paths import CACHE_DIR
from .llm_models import LLMModel
from .vicuna_models import VicunaModel
from .openai_models import OpenAIComplModel, OpenAIChatModel


# TODO think about putting both classes together;
class PlanningModel:

    def __init__(self, model_type: str, model_param: dict, examples_dict: dict, init_prompt: str):
        """

        :param model_type:
        :param model_param:
        :param examples_dict:
        :param init_prompt:
        """
        self.model_param = model_param
        self.examples_dict = examples_dict
        self.init_prompt = init_prompt
        self.examples_chat: bool = self.model_param.get('examples_chat', False)
        self.model: LLMModel = create_llm_model(model_type, model_param)
        self.initialize_model()

    def initialize_model(self):
        """
        Initializes the model by creating the initial prompt, initializing self.model with it and
        adding few-shot examples if specified
        :return:
        """
        self.model.init_model(self.init_prompt)
        if self.examples_chat:
            examples = self.get_list_examples()
            self.add_few_shot_dialogue_style(examples)

    def add_few_shot_dialogue_style(self, examples: List[dict]):
        """
        Adds the few_shot examples to the dialogue history
        :param examples:
        :return:
        """
        self.model.add_examples(examples)

    def get_initial_prompt(self) -> str:
        """
        Returns initial prompt of the LLMModel
        :return:
        """
        initial_prompt = self.model.get_initial_prompt()
        return initial_prompt

    def update_initial_prompt(self, new_init_prompt: str):
        """

        :param new_init_prompt:
        :return:
        """
        self.model.update_init_prompt(new_init_prompt=new_init_prompt)

    def get_history(self) -> List[dict]:
        """
        Returns the initial history, i.e. system prompt + few-shot examples, in the following format:
        [{"role": "system", "content": initial_prompt}, {"role": role, "content": content}, ...]
        :return:
        """
        history = self.model.get_history()
        return history

    def reset_history(self):
        """
        Resets the dialogue history to the initial_history
        :return:
        """
        self.model.reset_history()

    def generate(self, user_message):
        if not self.examples_chat:
            user_message = self.create_input_format_example(user_message)
        output = self.model.generate(user_message)
        return output


    def create_input_format_example(self, user_message: str) -> str:
        """
        Puts the user_message into the format that matches the few-shot examples if any provided
        e.g. f'Original: {user_message}\nOutput:'
        :param user_message:
        :return:
        """
        return f'{self.examples_dict["prefixes"]["input"]}\n{user_message}\n{self.examples_dict["prefixes"]["output"]}\n'

    def get_list_examples(self) -> List[dict]:

        examples = []

        if "pos_examples" in self.examples_dict.keys():
            for ex in self.examples_dict["pos_examples"]:
                input = ex['input']
                output = ex['output']
                examples.append({"role": "user", "content": input})
                examples.append({"role": "assistant", "content": output})

        return examples

class TranslationModel:

    def __init__(self, model_type: str, model_param: dict, examples_dict: dict, init_prompt: str):
        """

        :param model_type:
        :param model_param:
        :param examples_dict:
        :param init_prompt:
        """
        self.model_param = model_param
        self.examples_dict = examples_dict
        self.init_prompt = init_prompt
        self.examples_chat = self.model_param.get('examples_chat', False)
        self.model: LLMModel = create_llm_model(model_type, model_param)

        self.initialize_model()

    def initialize_model(self):
        """

        :return:
        """
        self.model.init_model(self.init_prompt)
        if self.examples_chat:
            examples = self.get_list_examples()
            self.add_few_shot_dialogue_style(examples)


    def add_few_shot_dialogue_style(self, examples: List[dict]):
        self.model.add_examples(examples)


    def create_input_format_example(self, user_message: str) -> str:
        """
        Puts the user_message into the format that matches the few-shot examples if any provided
        e.g. f'Original: {user_message}\nOutput:'
        Overwrite in subclass if format should be different
        :param user_message:
        :return:
        """
        return f'{self.examples_dict["prefixes"]["input"]}\n{user_message}\n{self.examples_dict["prefixes"]["output"]}\n'

    def generate(self, user_message: str):
        if not self.examples_chat:
            user_message = self.create_input_format_example(user_message)
        output = self.model.generate(user_message)
        return output

    def get_history(self):
        return self.model.get_history()

    def reset_history(self):
        self.model.reset_history()

    def get_initial_prompt(self):
        return self.model.get_initial_prompt()

    def update_initial_prompt(self, new_init_prompt: str):
        """

        :param new_init_prompt:
        :return:
        """
        self.model.update_init_prompt(new_init_prompt=new_init_prompt)

    def get_list_examples(self) -> List[dict]:

        examples = []

        if "pos_examples" in self.examples_dict.keys():
            for ex in self.examples_dict["pos_examples"]:
                input = ex['input']
                output = ex['output']
                examples.append({"role": "user", "content": input})
                examples.append({"role": "assistant", "content": output})

        if "neg_examples" in self.examples_dict.keys():
            for neg_ex in self.examples_dict["neg_examples"]:
                input = neg_ex['input']
                wrong_output = neg_ex['output']
                correct_output = neg_ex['wrong']
                examples.append({"role": "user", "content": input})
                examples.append({"role": "assistant", "content": wrong_output})
                examples.append(
                    {"role": "user", "content": f'This is wrong. The correct translation is: {correct_output}'})

        return examples




def create_llm_model(model_type: str, model_param: dict) -> LLMModel:
    """
    Creates different kinds of llm models of the subclasses of LLMModel (in llm_models.py)
    :param model_type: the name of the model to use
    :param model_param: dictionary with the parameters for the model
                        required keys: 'model_path', 'max_tokens', 'temp', 'max_history'
                        if vicuna model additionally: 'cuda_n', 'load_method'
    :return:
    """
    model_path = model_param['model_path']
    max_tokens = model_param.get('max_tokens', 512)
    temp = model_param.get('temp', 1.0)
    max_history = model_param.get('max_history', None)

    if model_type == 'openai_chat':
        cache_dir = CACHE_DIR / Path(f'../llm_caches/openai_chat_{model_path}')
        model = OpenAIChatModel(model_name=model_type,
                                model_path=model_path,
                                max_tokens=max_tokens,
                                temp=temp,
                                max_history=max_history,
                                cache_directory=cache_dir)

    elif model_type == 'openai_comp':
        cache_dir = CACHE_DIR / Path(f'../llm_caches/openai_comp_{model_path}')
        model = OpenAIComplModel(model_name=model_type,
                                model_path=model_path,
                                max_tokens=max_tokens,
                                temp=temp,
                                max_history=max_history,
                                cache_directory=cache_dir)

    elif model_type in ['vicuna', 'vicuna-x-gpt']:
        cache_dir = CACHE_DIR / Path(f'../llm_caches/{model_type}_{model_path}')
        model = VicunaModel(model_name=model_type,
                            model_path=model_path,
                            cuda_n=model_param['cuda_n'],
                            load_method=model_param['load_method'],
                            max_tokens=max_tokens,
                            temp=temp,
                            max_history=max_history,
                                cache_directory=cache_dir)
    else:
        raise NotImplementedError

    return model

