import abc
import json
import os
from typing import Union, List, Dict

import openai
from loguru import logger
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

SEQ2SEQ_MODELS = ["google/flan-t5-base", "allenai/tk-instruct-3b-def"]


class Predictor(abc.ABC):

    def __init__(self, env_config: dict, model_name: str, use_openai: bool, max_new_tokens: int, num_return_sequences: int,
                 temperature: float, do_sample: bool):
        self.env_config = env_config
        self.model_name = model_name
        self.use_openai = use_openai
        self.max_new_tokens = max_new_tokens
        self.num_return_sequences = num_return_sequences
        self.temperature = temperature  # TODO implement increasing temperature when LM comes to a halt
        self.do_sample = do_sample

        if self.use_openai:
            if os.getenv("OPENAI_API_KEY") is None:
                with open("tools/openai_key.json", "r") as file:
                    openai.api_key = json.load(file)["key"]
        else:
            if self.do_sample and temperature == 0:
                raise RuntimeError(f"A temperature of 0 is only supported with OpenAI or if do_sample=False, try "
                                   "slightly higher values than 0.")

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

            logger.info("Loading HuggingFace model...")

            # Some models use the text generation task, other the text2text task. For the latter, the seq2seq model
            # class is needed, for the former the causal LM class
            if self.model_name in SEQ2SEQ_MODELS:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.model_name
                )
            else:
                # Set pad_token_id manually to fix HuggingFace warning, see https://stackoverflow.com/a/71397707
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            logger.info("Loading complete")

        self.current_prompt_id = 0
        self.current_prompt_template = self.prompt_templates[self.current_prompt_id]

    @property
    @abc.abstractmethod
    def prompt_templates(self) -> Dict[int, str]:
        pass

    def set_prompt_template(self, prompt_template_id: int):
        try:
            self.current_prompt_template = self.prompt_templates[prompt_template_id]
        except KeyError:
            logger.error(f"There is no prompt with ID {prompt_template_id}")

    def get_prompt_templates(self) -> Dict[int, str]:
        return self.prompt_templates

    def convert_to_prompt(self, state) -> str:
        """
        Default method to create the prompt from the current state. Can be overwritten in subclass for
        prompts specific to a certain app.

        :param state: The current state of the app
        :return: The prompt for the LM as a string
        """
        return f"{self.current_prompt_template} {state}"

    def predict(self, prompt) -> List[Union[int, str]]:
        if self.use_openai:
            prediction = openai.Completion.create(
                model=self.model_name,
                prompt=prompt,
                max_tokens=self.max_new_tokens,
                temperature=self.temperature
            )

            raw_predictions = [prediction["choices"][0]["text"]]
        else:
            encoded_prompt = self.tokenizer(prompt, return_tensors="pt").input_ids

            encoded_model_prediction = self.model.generate(
                encoded_prompt,
                do_sample=self.do_sample,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=self.num_return_sequences,
                num_beams=self.num_return_sequences
            )

            # From the prediction, use only the predicted tokens (self.max_new_tokens), decode them, then discard
            # the batch dimension with "[0]"
            if self.model_name in SEQ2SEQ_MODELS:
                raw_predictions = self.tokenizer.batch_decode(encoded_model_prediction, skip_special_tokens=True)
            else:
                raw_predictions = self.tokenizer.batch_decode(encoded_model_prediction[:, -self.max_new_tokens:])

        possible_buttons_to_press = []

        for raw_pred in raw_predictions:
            try:
                button = int(raw_pred)
            except ValueError:
                possible_buttons_to_press.append(raw_pred)
            else:
                possible_buttons_to_press.append(button)

        return possible_buttons_to_press
