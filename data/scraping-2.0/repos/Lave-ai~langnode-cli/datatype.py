from openai import OpenAI
import google.generativeai as genai
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import List, Dict
import torch


class DataType:
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        for key in self.keys_to_compare:
            if self.__dict__[key] != other.__dict__[key]:
                return False
        return True


class PrimitiveType(DataType):
    pass


class ModelType(DataType):
    pass


class MessagesType(PrimitiveType):
    keys_to_compare = ["_data"]

    def __init__(self, data: List[Dict[str, str]]) -> None:
        super().__init__()
        self._data = data

    @property
    def data(self):
        return self._data


class StringType(PrimitiveType):
    keys_to_compare = ["_data"]

    def __init__(self, data: str) -> None:
        super().__init__()
        self._data = data

    @property
    def data(self):
        return self._data


class IntType(PrimitiveType):
    keys_to_compare = ["_data"]

    def __init__(self, data: int) -> None:
        super().__init__()
        self._data = data

    @property
    def data(self):
        return self._data


class FloatType(PrimitiveType):
    keys_to_compare = ["_data"]

    def __init__(self, data: float) -> None:
        super().__init__()
        self._data = data

    @property
    def data(self):
        return self._data


class OpenAIClientType(ModelType):
    keys_to_compare = ["api_key"]

    def __init__(self, api_key: StringType) -> None:
        super().__init__()
        self.api_key = api_key
        self._data = None

    @property
    def data(self):
        if self._data is None:
            self._data = OpenAI(api_key=self.api_key.data)
        return self._data


class GeminiModelType(ModelType):
    keys_to_compare = ["model_name", "api_key"]

    def __init__(self, model_name: StringType, api_key: StringType) -> None:
        super().__init__()
        self.model_name = model_name
        self.api_key = api_key
        self._data = None

    @property
    def data(self):
        if self._data is None:
            genai.configure(api_key=self.api_key.data)
            self._data = genai.GenerativeModel(self.model_name.data)
        return self._data


class BnBConfigType(ModelType):
    keys_to_compare = ["load_in"]

    def __init__(self, load_in: StringType) -> None:
        super().__init__()
        self.load_in = load_in
        self._data = None

    @property
    def data(self):
        if self._data is None:
            if self.load_in.data == "4bit":
                self._data = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )

            elif self.load_in.data == "8bit":
                self._data = BitsAndBytesConfig(load_in_8bit=True)

        return self._data


class HfAutoModelForCasualLMType(ModelType):
    keys_to_compare = ["base_model_id", "quantization_config"]

    def __init__(
        self, *, quantization_config: BnBConfigType, base_model_id: StringType
    ) -> None:
        super().__init__()
        self.base_model_id = base_model_id
        self.quantization_config = quantization_config
        self._data = None

    @property
    def data(self):
        if self._data is None:
            self._data = AutoModelForCausalLM.from_pretrained(
                self.base_model_id.data,
                quantization_config=self.quantization_config.data,
                device_map="auto",
            )
        return self._data


class HfAutoTokenizerType(ModelType):
    keys_to_compare = ["base_model_id"]

    def __init__(self, *, base_model_id: StringType) -> None:
        super().__init__()
        self.base_model_id = base_model_id
        self._data = None

    @property
    def data(self):
        if self._data is None:
            self._data = AutoTokenizer.from_pretrained(self.base_model_id.data)
        return self._data
