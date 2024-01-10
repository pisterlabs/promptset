from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    LangchainEmbedding,
    ServiceContext,
)
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts.prompts import SimpleInputPrompt
from llama_index.llm_predictor import LLMPredictor
from llama_index.prompts import PromptTemplate
import torch
from transformers import AutoModelForCausalLM

from align_system.utils import logging


log = logging.getLogger(__name__)

query_wrapper_prompt = SimpleInputPrompt(
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{query_str}\n\n### Response:"
)


class LlamaIndex:
    def __init__(self, domain_docs_dir=None, device="cuda", model_name="falcon", retrieval_enabled=True):  # noqa
        if retrieval_enabled and domain_docs_dir is None:
            raise RuntimeError(
                "'domain_docs_dir' argument must not be empty if "
                "'retrieval_enabled' is True")
        self.domain_docs_dir = domain_docs_dir
        self.device = device
        self.model_name = model_name
        self.retrieval_enabled = retrieval_enabled

    def load_model(self):
        if self.model_name == 'falcon':
            self.model_name = 'tiiuae/falcon-7b-instruct'

        if self.device == 'cuda':
            model_kwargs = {"torch_dtype": torch.float16,
                            "device_map": "auto"}
            predictor_kwargs = {"device_map": "auto"}
        else:
            model_kwargs = {}
            predictor_kwargs = {}

        self.embed_model = LangchainEmbedding(HuggingFaceEmbeddings())

        # FalconForCausalLM
        # https://github.com/huggingface/transformers/blob/0188739a74dca8a9cf3f646a9a417af7f136f1aa/src/transformers/models/falcon/convert_custom_code_checkpoint.py#L37
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **model_kwargs)

        self.hf_llm = HuggingFaceLLM(
            context_window=2048,
            max_new_tokens=256,
            # generate_kwargs={"temperature": 0.25, "do_sample": False},
            # "temperature" parameter is not used if "do_sample" is False
            generate_kwargs={"do_sample": False},
            query_wrapper_prompt=query_wrapper_prompt,
            tokenizer_name=self.model_name,
            model=model,
            tokenizer_kwargs={"max_length": 2048},
            tokenizer_outputs_to_remove=["token_type_ids"],
            **predictor_kwargs)

        self.service_context = ServiceContext.from_defaults(
            embed_model=self.embed_model,
            chunk_size=512,
            llm=self.hf_llm)

        if self.retrieval_enabled:
            documents = SimpleDirectoryReader(self.domain_docs_dir).load_data()
            new_index = VectorStoreIndex.from_documents(
                documents,
                service_context=self.service_context)

            # query with embed_model specified
            self.query_engine = new_index.as_query_engine(streaming=True)
        else:
            log.info("Retrieval disabled")
            self.query_engine = LLMPredictor(self.hf_llm)

        self.model_loaded = True

    def run_inference(self, prompt):
        if self.retrieval_enabled:
            return self.query_engine.query(prompt)
        else:
            bare_template = PromptTemplate("{query_str}")

            return self.query_engine.predict(bare_template, query_str=prompt)


    def __call__(self, sample, target_kdma_values, **kwargs):
        prompt = sample['scenario']
        if sample['state'] is not None:
            prompt += f'\n{sample["state"]}'
        prompt += f'\n{sample["probe"]}'
        
        prompt += '\nIdentify the integer index of the choice that best completes the request.\n'
        prompt += 'Choices:\n'
        
        choice_strs = [
            f'{idx}: {choice}'
            for idx, choice in enumerate(sample['choices'])
        ]
        
        prompt += '\n'.join(choice_strs)
        
        log_file = kwargs.get('log_file', None)
        if log_file:
            log_file.write(f'Prompt:\n{prompt}\n\n')
        
        choice = None
        n_fail = 0
        while choice is None:
            
            response = str(self.run_inference(prompt))
            
            if log_file:
                log_file.write(f'Response:\n{response}\n\n')
            
            # find the first integer in the response
            for char in response:
                try:
                    choice = int(char)
                    break
                except ValueError:
                    pass
            
            if choice is not None:
                if choice > len(sample['choices']) - 1:
                    choice = None
            
            if choice is None:
                n_fail += 1
            
            if n_fail > 10:
                return {
                    'choice': 0,
                    'info': f'Failed to find choice in response: {response}'
                }
        
        return {
            'choice': choice,
            'info': response
        }
