from langchain import HuggingFaceHub


class HuggingFaceConnector:
    """
    A connector class for the google/flan-t5-xl language model via the HuggingFaceHub API.

    This class provides an interface for interacting with the google/flan-t5-xl language model through the
    HuggingFaceHub API.
    The `llm` attribute is an instance of the `HuggingFaceHub` class, which is used to generate text based on prompts.

    Attributes:
        llm (HuggingFaceHub): An instance of the HuggingFaceHub class representing the google/flan-t5-xl language model.
    """

    def __init__(self):
        self.llm = HuggingFaceHub(
            repo_id="google/flan-t5-xxl",
            model_kwargs={"temperature": 0.7}
        )

    @property
    def get_llm(self):
        """
        Returns the HuggingFaceHub language model instance associated with this connector.

        Returns:
            HuggingFaceHub: An instance of the HuggingFaceHub class representing the google/flan-t5-xl language model.
        """
        return self.llm
