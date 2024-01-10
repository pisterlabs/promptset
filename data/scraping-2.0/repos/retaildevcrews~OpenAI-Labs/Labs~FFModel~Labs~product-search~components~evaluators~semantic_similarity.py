# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import List

import numpy as np

from ffmodel.components.base import BaseSolutionComponent
from ffmodel.data_models.base import ExperimentDataModel
from ffmodel.utils.openai import OpenAIConfig, get_embedding, initialize_openai


class Component(BaseSolutionComponent[ExperimentDataModel]):
    """
    Semantic Similarity evaluator evaluates the similarity between text generated completions that have been translated
    by a machine to the reference human text.

    The returned value is the cosine similarity between the embedding of the expected output and the generated output.
    The value ranges from 0 to 1, with 1 being an exact match.

    Args
    ----
    config (str):  Consists of OpenAI configurations - this should include an API key and/or endpoint
    embedding_model (str): Embedding model to leverage for experimentation. Default setting is `text-embedding-ada-002`
    """

    def _post_init(self):
        # Initialize the embedding model
        config_names = self.args.pop("config", {})
        self.openai_config = OpenAIConfig.from_dict(config_names)
        initialize_openai(self.openai_config)

        # set embedding model
        self.embedding_model = self.args.get("embedding_model", "text-embedding-ada-002")
        self.call_embedding_function = get_embedding

    def execute(self, data_model: ExperimentDataModel) -> ExperimentDataModel:
        """
        Executes the component for the given data model and returns an
        updated data model.
        """

        # calculate the embeddings for the expected output and the completions
        expected_embeddings = [
            self.call_embedding_function(e, self.embedding_model) for e in data_model.request.expected_output
        ]
        completion_embeddings = [
            self.call_embedding_function(c, self.embedding_model) for c in data_model.model_output.completions
        ]

        semantic_similarity = []
        for e in expected_embeddings:
            for c in completion_embeddings:
                semantic_similarity.append(self.get_semantic_similarity(e, c))

        results = {"semantic_similarity": semantic_similarity}
        data_model.experiment_metrics[self.get_id()] = results
        return data_model

    def get_semantic_similarity(self, ground_truth: List[float], prediction: List[float]) -> float:
        """
        Function used as helper in computing semantic similarity based on the embeddings

        Args
        -----
        ground_truth (List[float]): embedding of the ground truth
        prediction (List[float]): embedding of the predicted text

        Returns
        --------
        (float)
        """

        # generate mebeddings for ground truth and predicted prompt
        grd_truth_embed = np.asarray(ground_truth)
        pred_embed = np.asarray(prediction)

        # normalize feature embeddings
        norm_grd_truth = np.linalg.norm(grd_truth_embed)
        norm_pred = np.linalg.norm(pred_embed)
        grd_truth_embed = np.array(grd_truth_embed) / norm_grd_truth
        pred_embed = np.array(pred_embed) / norm_pred

        # compute cosine similarity
        cos_sim_ls = np.dot(grd_truth_embed, pred_embed)
        return cos_sim_ls
