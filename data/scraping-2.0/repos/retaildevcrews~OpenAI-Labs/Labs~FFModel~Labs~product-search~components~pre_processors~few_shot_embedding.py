# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import os
import pickle

import numpy as np

from ffmodel.components.base import BaseSolutionComponent
from ffmodel.data_models.base import InferenceDataModel, InferenceRequest, ModelState
from ffmodel.utils.openai import (
    OpenAIConfig,
    RetryParameters,
    get_embedding,
    initialize_openai,
)

REQUIRED_FIELDS = ["user_nl", "expected_output", "embedding"]


class Component(BaseSolutionComponent[InferenceDataModel[InferenceRequest, ModelState]]):
    """
    Few shot selection component based on embeddings from OpenAI models.

    For more information on the OpenAI embedding models, see: https://learn.microsoft.com/en-us/azure/cognitive-services/openai/concepts/understand-embeddings

    This class uses a pickle file to represent the embeddings for the few shot bank.
    Inside the pickle file (passed by `few_shot_file` config) should be a dictionary with the following structure:
    {
        "metadata": {
            "embedding_model": "embedding model used to generate the embeddings"
        },
        "data": [
            {
                "user_nl": "user_nl for the few shot",
                "expected_output": "expected output for the few shot",
                "embedding": "embedding for the few shot"
            },
            ...
        ]
    }

    Please use the static method create_few_shot_file to generate the pickle file

    Component Config args:
        - count: The number of few shots to select, defaults to 3
        - reverse: When true, the closest match is at the end, defaults to true
        - config: Dict[str, str], dictionary of config that control the OpenAI API
            - api_key_config_name: str, name of the config value to pull the api key from, defaults to OPENAI_API_KEY
            - api_endpoint_config_name: str, name of the config value to pull the api endpoint from, defaults to OPENAI_ENDPOINT
            - api_version: str, version of the OpenAI API to use, defaults to "2023-03-15-preview"
        - retry_params: Dict[str, Any], dictionary of retry parameters, with keys of:
            - tries: int, the maximum number of attempts. The first call counts as a try
            - delay: float, initial delay between attempts, in seconds
            - backoff: float, multiplier applied to the delay between attempts
            - max_delay: float, the maximum delay between attempts, in seconds
    Component Config supporting_data:
        - few_shot_file: Path to the pickle file containing the few shot examples.
        - cached_embeddings: Path to a pickle file containing prior embeddings, this follows the same format as the
        few_shot_file. The idea is to cache the embeddings for your evaluation data set and reuse them when rerunning the experiment.
    """

    call_embedding_function = get_embedding

    def get_embedding_with_cache(self, user_nl: str) -> list:
        """
        This function will return a cached embedding if a match is found for the given user_nl.
        Otherwise, it will call OpenAI to generate the embedding.

        Potential extension - we can add a max size for the cache, then append to it whenever we call the
        OpenAI embedding endpoint. This way we cache as we are running experiments. But, we would need
        to align on a strategy to which items to disregard from the cache (i.e., last item in the list gets dropped)
        """
        embedding = None

        if self.cached_embeddings:
            # search for a match based on the prompt
            embedding = self.cached_embeddings.get(user_nl, None)

        if embedding is None:
            initialize_openai(self.openai_config)
            embedding = Component.call_embedding_function(user_nl, self.embedding_model, self.retry_params)

        return embedding

    def _load_cached_embeddings(self, cache_file: str):
        data_file = None
        with open(cache_file, "rb") as f:
            # The pickled file has a dict with metadata and data that holds prompts
            # with their embeddings, thus, we only care for data.
            data_file = pickle.load(f)["data"]

        if data_file:
            self.cached_embeddings = {data["user_nl"]: data["embedding"] for data in data_file}

    def _post_init(self):
        # Loads the few shots
        few_shot_info = self.supporting_data.get("few_shot_file", None)
        if few_shot_info is None:
            raise ValueError("Argument 'few_shot_file' must be provided")

        self.few_shot_file = few_shot_info.file_path
        self._load_few_shots()

        # Loads the cached embeddings
        self.cached_embeddings = None
        cached_embeddings_config = self.supporting_data.get("cached_embeddings", None)
        if cached_embeddings_config:
            self._load_cached_embeddings(cached_embeddings_config.file_path)

        # Parse the input arguments
        config_names = self.args.pop("config", {})
        self.openai_config = OpenAIConfig.from_dict(config_names)

        retry_params = self.args.pop("retry_params", {})
        self.retry_params = RetryParameters.from_dict(retry_params)

        # Sets defaults for other values
        self.count = self.args.get("count", 3)
        self.reverse = self.args.get("reverse", True)

    def _load_few_shots(self):
        """
        Loads the few shots from the given file.

        Performs validation on each data point to make sure the required information is present
        """
        # Load the few shots - See "create_few_shot_file" for the contents
        with open(self.few_shot_file, "rb") as f:
            few_shot_bank = pickle.load(f)

        try:
            self.embedding_model = few_shot_bank["metadata"]["embedding_model"]
        except KeyError:
            raise ValueError("Few shot file does not contain metadata with embedding_model specified")

        # Validate data
        for d in few_shot_bank["data"]:
            for field in REQUIRED_FIELDS:
                if field not in d:
                    raise ValueError(f"Few shot data point missing required field {field}")

        self.few_shot_bank = few_shot_bank["data"]

        # Pull out the embeddings to a numpy array for faster cosine similarity calculation
        embeddings = []
        for item in self.few_shot_bank:
            normalization = np.linalg.norm(item["embedding"])
            embeddings.append(np.array(item["embedding"]) / normalization)

        self.embeddings = np.array(embeddings)

    def _get_cosine_sim(self, embedding: list):
        "get cosine similarities between few shot banks and the embedding"
        embedding = np.array(embedding) / np.linalg.norm(embedding)
        return np.dot(self.embeddings, embedding)

    def execute(
        self, data_model: InferenceDataModel[InferenceRequest, ModelState]
    ) -> InferenceDataModel[InferenceRequest, ModelState]:
        """
        Executes the component for the given data model and returns an
        updated data model.
        """
        prompt_text = data_model.request.user_nl
        prompt_embedding = self.get_embedding_with_cache(prompt_text)

        # calculate cosine similarity between prompt and few shot examples
        cos_sim_ls = self._get_cosine_sim(prompt_embedding)

        # find top n few shot examples using cosine similarity
        # Reverse is passed so that the closest match is first
        top_n_index = sorted(range(len(cos_sim_ls)), key=lambda i: cos_sim_ls[i], reverse=True)[: self.count]
        few_shots = [self.few_shot_bank[i] for i in top_n_index]

        if self.reverse:
            few_shots = list(reversed(few_shots))

        # Format for and add to the completion_pairs
        completion_pairs = [(few_shot["user_nl"], few_shot["expected_output"]) for few_shot in few_shots]
        data_model.state.completion_pairs.extend(completion_pairs)

        return data_model

    @staticmethod
    def create_few_shot_file(
        input_file: str,
        embedding_model: str,
        openai_config: OpenAIConfig = OpenAIConfig(),
        retry_params: RetryParameters = RetryParameters(),
        reporting_interval: int = 500,
    ) -> str:
        """Creates a few shot embedding file from the given dataset

        The dataset must contain the keys 'user_nl' and 'expected_output' and be jsonl format.
        Note: If expected_output is a list, the first element is used.

        The generated fewshot bank is a pickle file containing a dictionary with two fields:
            - metadata: Currently only key is the embedding model used
            - data: List of dictionaries containing the examples
        """
        # Load the dataset
        data = []
        with open(input_file, "r") as f:
            for line in f.readlines():
                data.append(json.loads(line))

        # Add the embeddings and select first expected output if a list
        initialize_openai(openai_config)
        print(f"Generating embeddings for {len(data)} points")
        for i, d in enumerate(data):
            if type(d["expected_output"]) is list:
                d["expected_output"] = d["expected_output"][0]
            d["embedding"] = Component.call_embedding_function(
                prompt=d["user_nl"],
                model=embedding_model,
                retry_parameters=retry_params,
            )

            if i % reporting_interval == 0:
                print(f"Completed {i+1} out of {len(data)} embeddings")

        few_shot_bank = {
            "metadata": {"embedding_model": embedding_model},
            "data": data,
        }
        output_file = f"{os.path.splitext(input_file)[0]}_{embedding_model}.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(few_shot_bank, f)

        print(f"Few shot pickle file generated and saved to {output_file}")

        return output_file
