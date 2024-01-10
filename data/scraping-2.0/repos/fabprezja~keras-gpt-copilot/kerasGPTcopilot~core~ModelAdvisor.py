# Author: Fabi Prezja <faprezja@fairn.fi>
# Copyright (C) 2023 Fabi Prezja
# License: MIT License (see LICENSE.txt for details)

import io
import json
import sys

import numpy as np
import openai
import pyperclip
from tensorflow.keras.models import Model

from kerasGPTcopilot.extras.util import CustomJSONEncoder


class ModelAdvisor:
    """
    A class to interact with OpenAI GPT to get suggestions on improving deep learning model performance.

    Attributes:
        model (Model): A Keras model object.
        history (History): A Keras History object.
        api_key (str): The API key for OpenAI.
        model_summary (bool, optional): Whether to include the model summary in the prompt. Defaults to False.
        model_config (bool, optional): Whether to include the model configuration in the prompt. Defaults to True.
        optimizer_info (bool, optional): Whether to include the optimizer information in the prompt. Defaults to True.
        add_info (str, optional): Additional information to include in the prompt. Defaults to None.
        test_loss (float, optional): Test loss value to include in the prompt. Defaults to None.
        test_metric (Union[float, Tuple[str, float]], optional): Test metric name and value. Defaults to None.
        un_trainable (bool, optional): Whether to include untrainable layer names in the prompt. Defaults to False.
        window_size (int, optional): The window size for smoothing the validation curves. Defaults to None.
        downsample_factor (int, optional): The downsampling factor for the validation curves. Defaults to None.
        suffix_text (str, optional): Additional text to include at the end of the prompt. Defaults to None.
        round_vc (int, optional): The number of decimal places to round the validation curves values. Defaults to 5.
    """

    def __init__(
            self, model: Model,
            history,
            api_key,
            model_summary=False,
            model_config=True,
            optimizer_info=True,
            add_info=None,
            test_loss=None,
            test_metric=None,
            un_trainable=False,
            window_size=None,
            downsample_factor=None,
            suffix_text=None,
            round_vc=5
    ):
        self.model = model
        self.history = history
        self.api_key = api_key
        self.model_summary = model_summary
        self.model_config = model_config
        self.optimizer_info = optimizer_info
        self.add_info = add_info
        self.test_loss = test_loss
        self.test_metric = test_metric
        openai.api_key = self.api_key
        self.un_trainable = un_trainable
        self.window_size = window_size
        self.downsample_factor = downsample_factor
        self.suffix_text = suffix_text
        self.round_vc = round_vc

    def _smooth_and_downsample(self, data):
        """
        Smooth and downsample the input data.

        Args:
            data (List[float]): The input data to be smoothed and downsampled.

        Returns:
            List[float]: The smoothed and downsampled data.
        """
        if self.downsample_factor:
            data = data[::self.downsample_factor]

        if self.window_size:
            data = np.convolve(data, np.ones((self.window_size,)) / self.window_size, mode='valid')

        return np.array(data).tolist()

    def _collect_validation_curves(self):
        """
        Collect the validation curves of the model's training history.

        Returns:
            Dict[str, List[float]]: The modified validation curves with smoothed and downsampled values.
        """
        original_history = self.history.history
        modified_history = {}

        for metric, data in original_history.items():
            modified_data = self._smooth_and_downsample(data)
            modified_history[metric] = [round(value, self.round_vc) for value in modified_data]

        return modified_history

    def _collect_model_info(self):
        """
        Collect various model information based on the user's requirements.

        Returns:
            Dict[str, Union[str, Dict]]: A dictionary containing the requested model information.
        """
        model_info = {}

        if self.model_summary:
            summary_str = io.StringIO()
            sys.stdout = summary_str
            self.model.summary()
            sys.stdout = sys.__stdout__
            model_info['summary'] = summary_str.getvalue()

        if self.model_config:
            model_info['config'] = self.model.get_config()

        if self.optimizer_info:
            model_info['optimizer'] = self.model.optimizer.get_config()

        if self.un_trainable:
            model_info['un_trainable'] = {layer.name: layer.trainable for layer in self.model.layers if
                                          not layer.trainable}

        return model_info

    def _format_prompt(self, validation_curves, model_info):
        """
        Format the input prompt for the GPT model.

        Args:
            validation_curves (Dict[str, List[float]]): The modified validation curves.
            model_info (Dict[str, Union[str, Dict]]): The collected model information.

        Returns:
            str: The formatted input prompt for the GPT model.
        """
        prompt = "I have a deep learning model with the following configuration:\n\n"

        if 'summary' in model_info:
            prompt += model_info['summary'] + "\n"

        if 'config' in model_info:
            prompt += "Model Configuration:\n" + json.dumps(model_info['config'], indent=2,
                                                            cls=CustomJSONEncoder) + "\n"

        if 'optimizer' in model_info:
            prompt += "Optimizer Configuration:\n" + json.dumps(model_info['optimizer'], indent=2,
                                                                cls=CustomJSONEncoder) + "\n"

        if 'un_trainable' in model_info:
            prompt += "Untrainable layers:\n" + json.dumps(model_info['un_trainable'], indent=2) + "\n"

        prompt += "The validation curves over epochs are:\n"
        prompt += json.dumps(validation_curves, indent=2) + "\n"

        if self.downsample_factor is not None:
            prompt += f"The validation curves epochs are downsampled by a factor of {self.downsample_factor}.\n"

        if self.test_loss is not None and self.test_metric is not None:
            metric_name, metric_value = self.test_metric if isinstance(self.test_metric, tuple) else (
                None, self.test_metric)
            prompt += f"The test loss is {self.test_loss} and the {metric_name} is {metric_value}.\n"

        if self.add_info is not None:
            prompt += self.add_info + "\n"

        prompt += "How can I improve the model performance based on this information?"

        if self.suffix_text is not None:
            prompt += "\n" + self.suffix_text

        return prompt

    def _get_suggestion_from_api(self, prompt, model="gpt-4", temperature=0.5, max_tokens=400):
        """
        Get a suggestion from the GPT model using the OpenAI API.

        Args:
            prompt (str): The input prompt for the GPT model.
            model (str, optional): The name of the OpenAI model to use. Defaults to "gpt-4".
            temperature (float, optional): The temperature to control the randomness of the output. Defaults to 0.5.
            max_tokens (int, optional): The maximum number of tokens in the output. Defaults to 400.

        Returns:
            str: The suggestion returned by the GPT model.
        """
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system",
                 "content": "You are an AI language model trained to give suggestions on how to improve deep learning "
                            "models."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )

        return response['choices'][0]['message']['content'].strip()

    def _save_to_file(self, prompt, suggestion, file_path):
        """
        Save the input prompt and suggestion to a file.

        Args:
            prompt (str): The input prompt for the GPT model.
            suggestion (str): The suggestion returned by the GPT model.
            file_path (str): The path to the file where the prompt and suggestion will be saved.
        """
        with open(file_path, 'w') as f:
            f.write("Input prompt:\n")
            f.write(prompt)
            f.write("\n\nSuggestion:\n")
            f.write(suggestion)

    def _copy_text_to_clipboard(self, text):
        """
        Copy the given text to the clipboard.

        Args:
            text (str): The text to be copied to the clipboard.
        """
        pyperclip.copy(text)

    def get_suggestions(self,
                        print_cp_response=True,
                        print_cp_prompt=True,
                        model="gpt-4",
                        temperature=0.5,
                        max_tokens=400,
                        save_to_file=None,
                        copy_to_clipboard=True,
                        cp_prompt_only=False):
        """
        Generate and return suggestions for improving the deep learning model.

        Args:
            print_cp_response (bool, optional): Whether to print the suggestion to the console. Defaults to True.
            print_cp_prompt (bool, optional): Whether to print the input prompt to the console. Defaults to True.
            model (str, optional): The name of the OpenAI model to use. Defaults to "gpt-4".
            temperature (float, optional): The temperature to control the randomness of the output. Defaults to 0.5.
            max_tokens (int, optional): The maximum number of tokens in the output. Defaults to 400.
            save_to_file (str, optional): The path to save the input prompt and suggestion to a file. Defaults to None.
            copy_to_clipboard (bool, optional):Copy the input prompt to the clipboard. Defaults to True.
            cp_prompt_only (bool, optional): Return only the copilot prompt without API suggestions. Defaults to False.

        Returns:
            str: The suggestion for improving the deep learning model.
        """
        validation_curves = self._collect_validation_curves()
        model_info = self._collect_model_info()

        prompt = self._format_prompt(validation_curves, model_info)

        if print_cp_prompt:
            print("Input prompt:\n", prompt)

        if copy_to_clipboard:
            self._copy_text_to_clipboard(prompt)

        if cp_prompt_only:
            return prompt

        suggestion = self._get_suggestion_from_api(prompt, model, temperature, max_tokens)

        if print_cp_response:
            print("Suggestion:\n", suggestion)

        if save_to_file:
            self._save_to_file(prompt, suggestion, save_to_file)

        return suggestion

    def get_follow_up_suggestion(self,
                                 initial_suggestion,
                                 follow_up_question,
                                 model="gpt-4",
                                 temperature=0.5,
                                 max_tokens=600,
                                 print_cp_response=True):
        """
        Generate and return a follow-up suggestion based on a follow-up question.

        Args:
            initial_suggestion (str): The initial suggestion.
            follow_up_question (str): The follow-up question.
            model (str): The name of the OpenAI model to use. Defaults to "gpt-4".
            temperature (float, optional): The temperature to control the randomness of the output. Defaults to 0.5.
            max_tokens (int, optional): The maximum number of tokens in the output. Defaults to 600.
            print_cp_response (bool, optional): Print the follow-up suggestion to the console. Defaults to True.

        Returns:
            str: The follow-up suggestion for improving the deep learning model.
        """
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system",
                 "content": "You are an AI language model trained to give suggestions on how to improve deep learning "
                            "models."},
                {"role": "user", "content": initial_suggestion},
                {"role": "user", "content": follow_up_question}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )

        follow_up_suggestion = response['choices'][0]['message']['content'].strip()

        if print_cp_response:
            print("Follow-up suggestion:\n", follow_up_suggestion)

        return follow_up_suggestion
