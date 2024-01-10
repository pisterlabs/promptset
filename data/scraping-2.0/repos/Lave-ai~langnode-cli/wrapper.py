from typing import Any
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import google.generativeai as genai
import torch
from openai import OpenAI
import csv
import os

class WrapperMixin:
    def __init__(self, built_instance) -> None:
        self.keys_to_compare = ['built_instance']
        self.built_instance = built_instance

    def __repr__(self) -> str:
        return __class__.__name__.replace("Wrapper", "")

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        self_dict_filtered = {key: self.__dict__[key] for key in self.keys_to_compare}
        other_dict_filtered = {key: other.__dict__[key] for key in self.keys_to_compare}

        return self_dict_filtered == other_dict_filtered


class CsvWriterWrapper(WrapperMixin):
    def __init__(self, *, csv_file_path, **columnname__value) -> None:
        self.keys_to_compare = ["csv_file_path"]
        self.csv_file_path = csv_file_path
        self.columnname__value = columnname__value

    def build(self):
        pass

    def __call__(self) -> None:
        file_exists = os.path.exists(self.csv_file_path) and os.path.getsize(self.csv_file_path) > 0

        data_row = {key: value.built_instance for key, value in self.columnname__value.items()}

        with open(self.csv_file_path, 'a' if file_exists else 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=self.columnname__value.keys())

            if not file_exists:
                writer.writeheader()

            writer.writerow(data_row)

    @classmethod
    def definition(cls):
        return {
            "type": cls.__name__.replace("Wrapper", ""),
            "fields": [
                {"name": "csv_file_path", "type": "str", "required": True, "default": None, "is_property": True},
            ]
        }


class ChatTemplateWrapper(WrapperMixin):
    def __init__(self, *, messages) -> None:
        self.keys_to_compare = ["messages"]
        self.messages = messages

    def build(self):
        for m in self.messages:
            assert "role" in m
            assert "content" in m
            assert m['role'] in ("user", "system", "assistant")

        self.built_instance = self.messages

    @classmethod
    def definition(cls):
        return {
            "type": cls.__name__.replace("Wrapper", ""),
            "fields": [
                {"name": "messages", "type": "array", "required": True, "default": None, "is_property": True},
            ]
        }

class OpenaiCompeletionWrapper(WrapperMixin):
    def __init__(self, client, prompt, *, model_name, max_tokens, temperature) -> None:
        self.keys_to_compare = ["client", "model_name", "max_tokens", "temperature", "prompt"]
        self.client = client
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.prompt = prompt

    def __call__(self) -> Any:
        chat_completion = self.client.built_instance.chat.completions.create(
            messages=self.prompt.built_instance,
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        print(chat_completion.choices[0].message.content)
        return WrapperMixin(chat_completion.choices[0].message.content)

    def build(self):
        pass

    @classmethod
    def definition(cls):
        return {
            "type": cls.__name__.replace("Wrapper", ""),
            "fields": [
                {"name": "client", "type": "OpenaiClient", "required": True, "default": None, "is_property": False},
                {"name": "model_name", "type": "str", "required": True, "default": None, "is_property": True},
                {"name": "max_tokens", "type": "int", "required": False, "default": 1024, "is_property": True},
                {"name": "temperature", "type": "float", "required": False, "default": 0.0, "is_property": True},
                {"name": "prompt", "type": "ChatTemplate", "required": True, "default": None, "is_property": False},
            ]
        }


class OpenaiClientWrapper(WrapperMixin):
    def __init__(self, *, api_key) -> None:
        self.keys_to_compare = ["api_key"]
        self.api_key = api_key if api_key else os.environ.get('LANGNODE_OPENAI_API_KEY')

    def build(self):
        self.built_instance = OpenAI(
            api_key=self.api_key)

    @classmethod
    def definition(cls):
        return {
            "type": cls.__name__.replace("Wrapper", ""),
            "fields": [
                {"name": "api_key", "type": "str", "required": True, "default": None, "is_property": True},
            ]
        }


class GeminiGeneratorWrapper(WrapperMixin):
    def __init__(self, model, prompt, *, max_output_tokens, temperature) -> None:
        self.keys_to_compare = ["model", "max_output_tokens", "temperature", "prompt"]
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.prompt = prompt

    def __call__(self) -> Any:
        responses = self.model.built_instance.generate_content(
            self.to_gemini_chat_components(),
        generation_config={
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
        },)
        print(responses.text)
        return WrapperMixin(responses.text)

    def to_gemini_chat_components(self):
        messages = []
        for p in self.prompt.built_instance:
            if p["role"] == "user":
                role = "user"
            elif p["role"] == "assistant":
                role = "model"
            else:
                role = "user"
            messages.append({"role": role, "parts": [p["content"]]})
        return messages

    def build(self):
        pass

    @classmethod
    def definition(cls):
        return {
            "type": cls.__name__.replace("Wrapper", ""),
            "fields": [
                {"name": "model", "type": "GeminiModel", "required": True, "default": None, "is_property": False},
                {"name": "max_output_tokens", "type": "int", "required": False, "default": 1024, "is_property": True},
                {"name": "temperature", "type": "float", "required": False, "default": 0.0, "is_property": True},
                {"name": "prompt", "type": "ChatTemplate", "required": True, "default": None, "is_property": False},
            ]
        }


class GeminiModelWrapper(WrapperMixin):
    def __init__(self, *, model_name, api_key) -> None:
        self.keys_to_compare = ["model_name", "api_key"]
        self.model_name = model_name
        self.api_key = api_key if api_key else os.environ.get('LANGNODE_GEMINI_API_KEY')

    def build(self):
        genai.configure(api_key=self.api_key)
        self.built_instance = genai.GenerativeModel(self.model_name)

    @classmethod
    def definition(cls):
        return {
            "type": cls.__name__.replace("Wrapper", ""),
            "fields": [
                {"name": "model_name", "type": "str", "required": True, "default": None, "is_property": True},
                {"name": "api_key", "type": "str", "required": True, "default": None, "is_property": True},
            ]
        }


class HfBitsAndBytesConfigWrapper(WrapperMixin):
    def __init__(self, *, load_in) -> None:
        self.keys_to_compare = ["load_in"]
        self.load_in = load_in

    def build(self):
        if self.load_in == "4bit":
            self.built_instance = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16
                )

        elif self.load_in == "8bit":
            self.built_instance = BitsAndBytesConfig(
                load_in_8bit=True)

    @classmethod
    def definition(cls):
        return {
            "type": cls.__name__.replace("Wrapper", ""),
            "fields": [
                {"name": "load_in", "type": "enum", "enum_values": ["4bit", "8bit"], "required": True, "default": "4bit", "is_property": True},
            ]
        }


class HfAutoModelForCasualLMWrapper(WrapperMixin):
    def __init__(self, quantization_config, *, base_model_id: str) -> None:
        self.keys_to_compare = ["base_model_id", "quantization_config"]
        self.base_model_id = base_model_id
        self.quantization_config = quantization_config

    def build(self):
        self.built_instance = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            quantization_config=self.quantization_config.built_instance,
            device_map="auto",
        )

    @classmethod
    def definition(cls):
        return {
            "type": cls.__name__.replace("Wrapper", ""),
            "fields": [
                {"name": "base_model_id", "type": "str", "required": True, "default": None, "is_property": True},
                {"name": "quantization_config", "type": "HfBitsAndBytesConfig", "required": True, "default": None, "is_property": False},
            ]
        }


class HfAutoTokenizerWrapper(WrapperMixin):
    def __init__(self, *, base_model_id: str) -> None:
        self.keys_to_compare = ["base_model_id"]
        self.base_model_id = base_model_id

    def build(self):
        self.built_instance = AutoTokenizer.from_pretrained(self.base_model_id)

    @classmethod
    def definition(cls):
        return {
            "type": cls.__name__.replace("Wrapper", ""),
            "fields": [
                {"name": "base_model_id", "type": "str", "required": True, "default": None, "is_property": True}
            ]
        }


class HfModelGeneratorWrapper(WrapperMixin):
    def __init__(self, model, tokenizer, prompt, *, temperature, max_new_tokens, repetition_penalty) -> None:
        self.keys_to_compare = ['model', 'tokenizer', 'temperature', 'max_new_tokens', 'repetition_penalty', 'prompt']
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.repetition_penalty = repetition_penalty
        self.prompt = prompt

    def __call__(self):
        self.model.built_instance.eval()
        model_input = self.tokenizer.built_instance.apply_chat_template(self.prompt.built_instance,
                                                                        return_tensors="pt",
                                                                        add_generation_prompt=True).to("cuda")
        num_input_tokens = model_input.shape[1]
        generated_tokens = self.model.built_instance.generate(model_input,
                                               max_new_tokens=self.max_new_tokens,
                                               repetition_penalty=1.15)

        output_tokens = generated_tokens[:, num_input_tokens:]
        assistant_response = self.tokenizer.built_instance.decode(output_tokens[0], skip_special_tokens=True)

        print(assistant_response)

        return WrapperMixin(assistant_response)

    def build(self):
        pass

    @classmethod
    def definition(cls):
        return {
            "type": cls.__name__.replace("Wrapper", ""),
            "fields": [
                {"name": "model", "type": "HfAutoModelForCasualLM", "required": True, "default": None, "is_property": False},
                {"name": "tokenizer", "type": "HfBitsAndBytesConfig", "required": True, "default": None, "is_property": False},
                {"name": "temperature", "type": "float", "required": False, "default": 0.0, "is_property": True},
                {"name": "max_new_tokens", "type": "int", "required": False, "default": 1024, "is_property": True},
                {"name": "repetition_penalty", "type": "float", "required": False, "default": 0.0, "is_property": True},
                {"name": "prompt", "type": "ChatTemplate", "required": True, "default": None, "is_property": False},
            ]
        }
