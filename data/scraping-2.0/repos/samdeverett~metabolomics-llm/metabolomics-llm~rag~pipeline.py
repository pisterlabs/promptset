"""
This script provides functionality for initializing a LLM to be used in RAG.
"""

import transformers

from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from torch import cuda, bfloat16
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers.pipelines.text_generation import TextGenerationPipeline


def init_model(model_id: str, hf_auth_token: str) -> PreTrainedModel:
    """Initializes a pretrained Hugging Face model.

    Args:
        model_id: The model id of a pretrained model configuration hosted inside a model repo on huggingface.co.
            Valid model ids can be located at the root-level, like bert-base-uncased,
            or namespaced under a user or organization name, like dbmdz/bert-base-german-cased.
        hf_auth_token: Hugging Face access token (https://huggingface.co/settings/tokens).
    """
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    
    # set quantization configuration to load large model with less GPU memory
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )

    # initialize Hugging Face model
    model_config = transformers.AutoConfig.from_pretrained(
        model_id,
        use_auth_token=hf_auth_token
    )

    # pull model in evaluation (not training) mode
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
        use_auth_token=hf_auth_token
    )
    model.eval()

    return model


def init_tokenizer(model_id: str, hf_auth_token: str) -> PreTrainedTokenizerBase:
    """Initializes a pretrained Hugging Face tokenizer.

    Args:
        model_id: The model id of a pretrained tokenizer hosted inside a model repo on huggingface.co.
            Valid model ids can be located at the root-level, like bert-base-uncased,
            or namespaced under a user or organization name, like dbmdz/bert-base-german-cased.
        hf_auth_token: Hugging Face access token (https://huggingface.co/settings/tokens).
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        use_auth_token=hf_auth_token
    )
    return tokenizer


def init_text_generation_pipeline(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        # GenerationConfig parameters (https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig)
        max_new_tokens: int = 4096,
        do_sample: bool = True,
        temperature: float = 0.6,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1
    ) -> TextGenerationPipeline:
    """Initializes a text generation pipeline for predicting the words that follow a prompt.

    Args:
        model: The model that will be used by the pipeline to make predictions.
        tokenizer: The tokenizer that will be used by the pipeline to encode data for the model.
        max_new_tokens: The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.
        do_sample: Whether or not to use sampling; use greedy decoding otherwise.
        temperature: The value used to modulate the next token probabilities.
        top_p: If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
        repetition_penalty: The parameter for repetition penalty. 1.0 means no penalty. See this paper for more details: https://arxiv.org/pdf/1909.05858.pdf.
    """
    text_generation_pipeline = transformers.pipeline(
        task='text-generation',
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty
    )
    return text_generation_pipeline


def init_langchain_pipeline(pipeline: TextGenerationPipeline) -> HuggingFacePipeline:
    """Initializes a LangChain model for a given Hugging Face TextGenerationPipeline.
    
    Args:
        pipeline: Hugging Face TextGenerationPipeline to turn into a LangChain model.
    """
    pipeline = HuggingFacePipeline(pipeline=pipeline)
    return pipeline
