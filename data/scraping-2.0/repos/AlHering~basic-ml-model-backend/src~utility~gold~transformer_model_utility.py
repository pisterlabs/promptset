# -*- coding: utf-8 -*-
"""
****************************************************
*                     Utility                      *
*            (c) 2023 Alexander Hering             *
****************************************************
"""
import os
from datasets import load_dataset
from typing import Any, Optional, List, Union
from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModel
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding
import evaluate
import torch.nn.functional as F
from torch import Tensor
from langchain.llms import LlamaCpp
import pandas as pd
import numpy as np
from src.configuration import configuration as cfg


"""
MODEL INSTANTIATION: Loader classes
"""


class LanguageModel(ABC):
    """
    Abstract language model class.
    """

    @abstractmethod
    def generate(self, prompt: Union[str, List[str]]) -> Optional[Any]:
        """
        Main handler method for wrapping language model capabilities.
        :param prompt: User prompt(s).
        :return: Response(s), if generation was successful.
        """
        pass

    @abstractmethod
    def get_model_instance(self) -> Any:
        """
        Method for getting model instance.
        :return: LLM instance.
        """
        pass


class LlamaCppLM(LanguageModel):
    """
    General LM class for LlamaCpp.
    """

    def __init__(self, model_path: str, model_config: dict) -> None:
        """
        Initiation method.
        :param model_path: Relative model path.
        :param model_config: Model configuration.
        :param representation: Language model representation.
        """
        self.llm = LlamaCpp(
            model_path=os.path.join(
                cfg.PATHS.TEXTGENERATION_MODEL_PATH, model_path, model_config["model_version"]),
            **model_config.get("loader_kwargs", {})
        )

    def generate(self, prompt: Union[str, List[str]]) -> Optional[Any]:
        """
        Main handler method for wrapping language model capabilities.
        :param prompt: User prompt(s).
        :return: Response(s), if generation was successful.
        """
        return self.llm.generate(prompt if isinstance(prompt, list) else [prompt])

    def get_model_instance(self) -> Any:
        """
        Method for getting model instance.
        :return: LLM instance.
        """
        return self.llm


class AutoGPTQLM(LanguageModel):
    """
    General LM class for AutoGPTQ models.
    """

    def __init__(self, model_path: str, model_config: dict) -> None:
        """
        Initiation method.
        :param model_path: Relative model path.
        :param model_config: Model configuration.
        :param representation: Language model representation.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_path,
            local_files_only=True,
            **model_config.get("loader_kwargs", {}).get("tokenizer", {})
        )
        if "quantize_config" in model_config.get("loader_kwargs", {}).get("model", {}):
            if isinstance(model_config["loader_kwargs"]["model"]["quantize_config"], dict):
                model_config["loader_kwargs"]["model"]["quantize_config"] = BaseQuantizeConfig(
                    **model_config["loader_kwargs"]["model"]["quantize_config"]
                )
            self.model = AutoGPTQForCausalLM.from_quantized(
                pretrained_model_name_or_path=model_path,
                local_files_only=True,
                **model_config.get("loader_kwargs", {}).get("model", {})
            )
        else:
            self.model = AutoGPTQForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_path,
                local_files_only=True,
                **model_config.get("loader_kwargs", {}).get("model", {})
            )

    def generate(self, prompt: Union[str, List[str]]) -> Optional[Any]:
        """
        Main handler method for wrapping language model capabilities.
        :param prompt: User prompt(s).
        :return: Response(s), if generation was successful.
        """
        if isinstance(prompt, list):
            responses = []
            for single_prompt in prompt:
                inputs = self.tokenizer(single_prompt, return_tensors="pt")
                responses.append(self.model(**inputs))
            return responses
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            return self.model(**inputs)

    def get_model_instance(self) -> Any:
        """
        Method for getting model instance.
        :return: LLM instance.
        """
        return self.tokenizer, self.model


class LocalHFLM(LanguageModel):
    """
    General LM class for local Huggingface models.
    """

    def __init__(self, model_path: str, model_config: dict) -> None:
        """
        Initiation method.
        :param model_path: Relative model path.
        :param model_config: Model configuration.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_path,
            local_files_only=True,
            **model_config.get("loader_kwargs", {}).get("tokenizer", {})
        )
        self.model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=model_path,
            local_files_only=True,
            **model_config.get("loader_kwargs", {}).get("model", {})
        )

    def generate(self, prompt: Union[str, List[str]]) -> Optional[Any]:
        """
        Main embedding method.
        :param prompt: User prompt(s).
        :return: Response(s), if generation was successful.
        """
        if isinstance(prompt, list):
            responses = []
            for single_prompt in prompt:
                inputs = self.tokenizer(single_prompt, return_tensors="pt")
                responses.append(self.model(**inputs))
            return responses
        else:
            inputs = self.tokenizer(prompt, return_tensors="pt")
            return self.model(**inputs)

    def get_model_instance(self) -> Any:
        """
        Method for getting model instance.
        :return: LLM instance.
        """
        return self.tokenizer, self.model


class LocalHFEmbeddingLM(LocalHFLM):
    """
    General LM class for local Huggingface models for embedding.
    """

    def generate(self, prompt: Union[str, List[str]]) -> Optional[Any]:
        """
        Main embedding method.
        :param prompt: User prompt(s).
        :return: Response(s), if generation was successful.
        """
        if isinstance(prompt, list):
            responses = []
            for single_prompt in prompt:
                inputs = self.tokenizer(prompt, max_length=512,
                                        padding=True, truncation=True, return_tensors='pt')

                outputs = self.model(**inputs)
                embeddings = self.average_pool(outputs.last_hidden_state,
                                               inputs['attention_mask'])

                # normalize embeddings
                embeddings = F.normalize(embeddings, p=2, dim=1)
                responses.append(embeddings.tolist())
            return responses
        else:
            inputs = self.tokenizer(prompt, max_length=512,
                                    padding=True, truncation=True, return_tensors='pt')

            outputs = self.model(**inputs)
            embeddings = self.average_pool(outputs.last_hidden_state,
                                           inputs['attention_mask'])

            # normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            return embeddings.tolist()

    def average_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Average pooling function, taken from https://huggingface.co/intfloat/e5-large-v2.
        """
        last_hidden = last_hidden_states.masked_fill(
            ~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def get_model_instance(self) -> Any:
        """
        Method for getting model instance.
        :return: LLM instance.
        """
        return self.tokenizer, self.model


"""
MODEL INSTANTIATION: Parameter gateways
"""


"""
MODEL INSTANTIATION: Parameterized Language Models
"""
SUPPORTED_TYPES = {
    "llamacpp": {
        "loaders": {
            "_default": LlamaCppLM
        },
        "gateways": {}
    },
    "gptq": {
        "loaders": {
            "_default": AutoGPTQLM,
        },
        "gateways": {}
    },
    "openai": {
        "loaders": {
            "_default": None
        },
        "gateways": {}
    },
    "gpt4all": {
        "loaders": {
            "_default": None
        },
        "gateways": {}
    },
    "bedrock": {
        "loaders": {
            "_default": None
        },
        "gateways": {}
    },
    "cohere": {
        "loaders": {
            "_default": None
        },
        "gateways": {}
    },
    "google_palm": {
        "loaders": {
            "_default": None
        },
        "gateways": {}
    },
    "huggingface": {
        "loaders": {
            "_default": LocalHFLM,
            "embedding": LocalHFEmbeddingLM
        },
        "gateways": {}
    },
    "koboldai": {
        "loaders": {
            "_default": None
        },
        "gateways": {}
    },
    "mosaicml": {
        "loaders": {
            "_default": None
        },
        "gateways": {}
    },
    "replicate": {
        "loaders": {
            "_default": None
        },
        "gateways": {}
    },
    "anthropic": {
        "loaders": {
            "_default": None
        },
        "gateways": {}
    },
    "openllm": {
        "loaders": {
            "_default": None
        },
        "gateways": {}
    },
    "openlm": {
        "loaders": {
            "_default": None
        },
        "gateways": {}
    },
    "rwkv": {
        "loaders": {
            "_default": None
        },
        "gateways": {}
    }

}


def spawn_language_model_instance(model_path: str, model_config: dict) -> Optional[LanguageModel]:
    """
    Function for spawning language model instance based on configuration.
        :param model_path: Relative model path.
        :param model_config: Model configuration.
    :return: Language model instance if configuration was successful else None.
    """
    lm = SUPPORTED_TYPES.get(model_config.get("type"), {}).get(
        "loaders", {}).get(model_config.get("loader", "_default"))
    if lm is not None:
        lm = lm(model_path, model_config)
    return lm


"""
MODEL FINETUNING: Huggingface finetuning
"""


def load_dataset(dataset_type: str, dataset_target: str, **reading_kwargs: Optional[Any]) -> Optional[pd.DataFrame]:
    """
    Function for loading a dataset.
    :param dataset_type: Type of the dataset.
    :param dataset_target: Dataset target.
    :param reading_kwargs: Arbitrary keyword arguments for reading in data.
    :return: Dataframe.
    """
    reading_function_name = f"read_{dataset_type.lower()}"
    if hasattr(pd, reading_function_name):
        return getattr(pd, reading_function_name)(dataset_target, **reading_kwargs)


def hf_finetune_model(base_model_type: str, base_model: str,
                      ft_dataset_type: str, ft_dataset: str,
                      x_column: str, y_column: str, evaluation_metric: str,
                      output: str) -> None:
    """
    Function for finetuning Huggingface models for binay text classification.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=base_model,
        local_files_only=True if base_model_type == "local" else False)
    model = AutoModel.from_pretrained(
        pretrained_model_name_or_path=base_model,
        local_files_only=True if base_model_type == "local" else False)

    dataset = load_dataset(ft_dataset_type, ft_dataset)

    def tokenize_function(examples):
        return tokenizer(examples[x_column], padding="max_length", truncation=True)

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer)

    training_args = TrainingArguments(output_dir=output)
    metric = evaluate.load(evaluation_metric)

    def compute_metrics(eval_pred) -> Optional[dict]:
        """
        Function for computing metrics based on evaluation prediction.
        :param eval_pred: Evaluation prediction.
        :return: Evaluation results as dictionary.
        """
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    training_args = TrainingArguments(
        output_dir="test_trainer", evaluation_strategy="epoch")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["evaluation"],
        compute_metrics=compute_metrics,
    )
    trainer.train()
