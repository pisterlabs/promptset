"""Cluster module."""

import openai
import pandas as pd
from bertopic.representation import OpenAI
from InstructorEmbedding import INSTRUCTOR
from bertopic import BERTopic
from bertopic.backend import BaseEmbedder


class CustomEmbedder(BaseEmbedder):
    """Custom Embedder."""

    def __init__(
        self, embedding_model: str = "hkunlp/instructor-large", instruction: str = ""
    ):
        super().__init__()
        if not isinstance(instruction, str) or len(instruction) < 1:
            raise ValueError("`instruction` is required.")

        self.instruction = instruction
        self.embedding_model = INSTRUCTOR(embedding_model)

    def embed(self, documents, instruction, verbose=False):
        """Embed a list of documents into vectors."""
        instruction_documents = [[self.instruction, d] for d in documents]
        embeddings = self.embedding_model.encode(
            instruction_documents, show_progress_bar=verbose
        )
        return embeddings


INSTRUCTION = "Represent the intent of the user journey for users of a website"

PROMPT = """
I have a set of users that have the following page visit journey through our website.
The journeys are in the following format:
<Page Title A> -> <Page Title C> -> <Page Title N>

Here are the journeys:
[DOCUMENTS]

The pages visited have these keyword themes: [KEYWORDS]

Based on the information above, extract a short topic label that indicates the most likely persona (who is looking for this information) and intent of the users in the following format:
topic: <topic label>
"""


def analyze_clusters(
    df: pd.DataFrame,
    model: str = "gpt-3.5-turbo",
    api_key: str = None,
    min_topic_size: int = 100,
) -> None:
    """Analyze clusters.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with conversion data.
    model : str, optional
        The model to use for topic modeling, by default "gpt-3.5-turbo".
    api_key : str, optional
        The OpenAI API key, by default None.
    min_topic_size : int, optional
        The minimum topic size, by default 100.

    Returns
    -------
    None
    """
    # Check that key is present
    if api_key is None:
        raise ValueError("`api_key` is required.")

    # Set key
    openai.api_key = api_key

    representation_model = OpenAI(
        model=model,
        delay_in_seconds=5,
        exponential_backoff=True,
        diversity=0.5,
        prompt=PROMPT,
        chat=True,
    )
    embedding_model = CustomEmbedder(instruction=INSTRUCTION)

    # This is here if developers want to assign topic back to users
    users = df.user_id.tolist()
    docs = df.activity_list_text.tolist()

    topic_model = BERTopic(
        nr_topics="auto",
        embedding_model=embedding_model,
        representation_model=representation_model,
        min_topic_size=min_topic_size,
        verbose=True,
    )

    _, _ = topic_model.fit_transform(docs)

    topic_model.get_topic_info()

    topic_model.visualize_topics()

    return topic_model


# Path: cluster.py
