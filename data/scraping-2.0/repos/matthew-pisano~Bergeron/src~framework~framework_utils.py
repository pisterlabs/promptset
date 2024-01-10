import torch
from transformers import PreTrainedModel, AutoModelForCausalLM, PreTrainedTokenizer, AutoTokenizer, AutoModel

from src.logger import root_logger
from src.wrappers.hf_api_model import HFAPIModel
from src.wrappers.dev_model import DevModel
from src.wrappers.mock_tokenizer import MockTokenizer
from src.wrappers.openai_api_model import OpenAIAPIModel
from src.utils import ModelSrc


class ModelInfo:

    def __init__(self, pretrained_model_name_or_path: str, model_src: ModelSrc, model_class: PreTrainedModel | None = AutoModelForCausalLM, tokenizer_class: PreTrainedTokenizer | None = AutoTokenizer, model_task: str = None):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.model_src = model_src
        self.model_class = model_class
        self.tokenizer_class = tokenizer_class
        self.model_task = model_task

        if self.model_src == ModelSrc.HF_API and self.model_task is None:
            raise ValueError("A model task is required to use HuggingFace models")

    def as_dict(self):
        return vars(self)


def from_pretrained(model_info: ModelInfo) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Gets the pretrained model and tokenizer from the given model information

    Args:
        model_info: The model and tokenizer information to get the pretrained model and tokenizer
    Returns:
        A transformers pretrained model and tokenizer for usage within the framework"""

    root_logger.debug(f"Loading a pretrained model {model_info.pretrained_model_name_or_path} from {model_info.model_src}")
    if model_info.model_src == ModelSrc.OPENAI_API:
        return OpenAIAPIModel(model_info.pretrained_model_name_or_path), MockTokenizer(model_info.pretrained_model_name_or_path)
    elif model_info.model_src == ModelSrc.HF_API:
        return HFAPIModel(model_info.pretrained_model_name_or_path, model_info.model_task), MockTokenizer(model_info.pretrained_model_name_or_path)
    elif model_info.model_src == ModelSrc.DEV:
        return DevModel(model_info.pretrained_model_name_or_path), MockTokenizer(model_info.pretrained_model_name_or_path)
    else:
        try:
            model = model_info.model_class.from_pretrained(model_info.pretrained_model_name_or_path, torch_dtype=torch.bfloat16)
        except ValueError as e:
            root_logger.warning(f"Could not load {model_info.pretrained_model_name_or_path} as a {model_info.model_class} model.  Using AutoModel instead.")
            model = AutoModel.from_pretrained(model_info.pretrained_model_name_or_path, torch_dtype=torch.bfloat16)

        return model, model_info.tokenizer_class.from_pretrained(model_info.pretrained_model_name_or_path)
