import warnings
from typing import List

import openai

from opentrain.schemas import FineTune

warnings.simplefilter("once", category=UserWarning)


class Inference:
    """The `Inference` class is a wrapper around OpenAI's Completion API, making it easy
    to generate completions for any given prompt, using an OpenAI fine-tuned model.

    Args:
        model: the name of the OpenAI model to use for the inference.

    Attributes:
        model: the name of the OpenAI model to use for the inference.

    Examples:
        >>> from opentrain import Inference
        >>> inference = Inference(model="curie:ft-personal-<DATE>")
        >>> inference(prompt="This is a sample prompt.")
        'This is a sample completion.'
    """

    def __init__(self, model: str) -> None:
        """Initializes the `Inference` class.

        Args:
            model: the name of the OpenAI model to use for the inference.
        """
        self.model = model

    def __call__(self, prompt: str, **kwargs) -> str:
        """Generates the completion for a given prompt.

        Args:
            prompt: the prompt to generate the completion for. Should be aligned
                with the one used/defined for the fine-tuning, if applicable.
            **kwargs: the keyword arguments to pass to the OpenAI API. See
                https://platform.openai.com/docs/api-reference/completions/create.

        Returns:
            The completion for the given prompt.
        """
        kwargs.setdefault("temperature", 0.0)
        if kwargs["temperature"] != 0:
            warnings.warn(
                f"The `temperature` parameter is set to {kwargs['temperature']},"
                " instead of 0. That means the completion for a given prompt will be"
                " random, so the suggestion on fine-tuned models, unless desired"
                " otherwise, is to set the `temperature` to 0 so that the model is"
                " almost deterministic.",
                UserWarning,
                stacklevel=2,
            )
        response = openai.Completion.create(
            model=self.model,
            prompt=prompt,
            **kwargs,
        )
        return response.choices[0].text

    @classmethod
    def from_fine_tune_id(cls, fine_tune_id: str) -> "Inference":
        """Returns an `Inference` object from an OpenAI fine-tune ID.

        Args:
            fine_tune_id: the ID of the OpenAI fine-tune to use for the inference.

        Returns:
            An `Inference` object.
        """
        model = openai.FineTune.retrieve(fine_tune_id).fine_tuned_model
        if model is None:
            raise ValueError(
                "The model is not fine-tuned yet! Please wait a few minutes and try"
                " again."
            )
        return cls(model=model)


def list_fine_tunes() -> List[FineTune]:
    """List all fine-tuned models in your OpenAI account.

    Returns:
        A list of OpenAI fine-tunes, as `FineTune` objects.
    """
    return [FineTune(**fine_tune) for fine_tune in openai.FineTune.list()["data"]]
