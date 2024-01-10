"""HuggingFace Hub Client."""
import logging
from dataclasses import dataclass
# TODO: Replace langchain with guidance https://github.com/guidance-ai/guidance
from langchain import HuggingFaceHub, LLMChain, PromptTemplate
import numpy as np
from oaklib.utilities.apikey_manager import get_apikey_value
from sentence_transformers import SentenceTransformer

# Note: See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads
# for all relevant models

# TODO: We want to support querying a self-hosted HuggingFace LLM.


@dataclass
class HFHubClient:
    """A client for the HuggingFace Hub API."""

    def __init__(self):
        self.api_key = None
        self.models_by_name = {}
        self.modelnames_by_model = {}
        self.sentence_transformers_by_name = {}

    def __post_init__(self):
        try:
            self.api_key = get_apikey_value("hfhub-key")
        except ValueError:
            logging.info("HuggingFace Hub API key not found. Using models locally.")

    def get_model(self, modelname: str) -> HuggingFaceHub:
        """Retrieve a model from the Hub, given its repository name.

        Returns a model object of type
        langchain.llms.huggingface_hub.HuggingFaceHub
        """
        model = HuggingFaceHub(
            repo_id=modelname,
            verbose=True,
            model_kwargs={"temperature": 0.2, "max_length": 500},
            huggingfacehub_api_token=self.api_key,
            task="text-generation"
        )
        self.models_by_name[modelname] = model
        self.modelnames_by_model[model] = modelname
        self.sentence_transformers_by_name[modelname] \
                = SentenceTransformer(modelname)
        return model


    def query_hf_model(self, llm, prompt_text):
        """Interact with a GPT4All model."""
        logging.info(f"Complete: prompt[{len(prompt_text)}]={prompt_text[0:100]}...")

        template = """{prompt_text}"""

        prompt = PromptTemplate(template=template, input_variables=["prompt_text"])

        llm_chain = LLMChain(prompt=prompt, llm=llm)

        try:
            raw_output = llm_chain.run({"prompt_text": prompt_text})
        except ValueError as e:
            logging.error(e)
            raw_output = ""

        return raw_output

    def embeddings(self, text: str, modelname: str = None):
        if modelname is None:
            modelname = 'distilroberta-base'
        if modelname not in self.sentence_transformers_by_name:
            self.get_model(modelname)
        sentence_transformer = sentence_transformers_by_name[modelname]
        embedding = sentence_transformer.encode(text)
        return embedding

    def similarity(self, text1: str, text2: str, **kwargs):
        a1 = self.embeddings(text1, **kwargs)
        a2 = self.embeddings(text2, **kwargs)
        logger.debug(f"similarity: {a1[0:10]}... x {a2[0:10]}... // ({len(a1)} x {len(a2)})")
        return np.dot(a1, a2) / (np.linalg.norm(a1) * np.linalg.norm(a2))

    def euclidean_distance(self, text1: str, text2: str, **kwargs):
        a1 = self.embeddings(text1, **kwargs)
        a2 = self.embeddings(text2, **kwargs)
        return np.linalg.norm(np.array(a1) - np.array(a2))

