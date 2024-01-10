import openai
from enum import Enum
import os

class ProfanityThreshold(Enum):
    open = 1
    tolerant = 2
    strict = 3


class ProfaneException(Exception):
    pass


def is_profane(text: str, toxic_threshold: float = -0.355) -> int:
    """
    Check if text contains profanity.
    :param text: The text to check.
    :param toxic_threshold: This is the probability at which we evaluate that a "2" is likely real
    vs. should be discarded as a false positive
    :return: True if text contains profanity, False otherwise.
    """
    openai.api_base = "https://api.openai.com/v1"
    openai.api_key = os.environ.get("OPENAI_KEY")
    openai.organization = os.environ.get("OPENAI_ORG")
    response = openai.Completion.create(
        engine="content-filter-alpha",
        prompt="<|endoftext|>" + text + "\n--\nLabel:",
        temperature=0,
        max_tokens=1,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        logprobs=10,
    )
    output_label = response["choices"][0]["text"]

    if output_label == "2":
        # If the model returns "2", return its confidence in 2 or other output-labels
        logprobs = response["choices"][0]["logprobs"]["top_logprobs"][0]

        # If the model is not sufficiently confident in "2",
        # choose the most probable of "0" or "1"
        # Guaranteed to have a confidence for 2 since this was the selected token.
        if logprobs["2"] < toxic_threshold:
            logprob_0 = logprobs.get("0", None)
            logprob_1 = logprobs.get("1", None)

            # If both "0" and "1" have probabilities, set the output label
            # to whichever is most probable
            if logprob_0 is not None and logprob_1 is not None:
                if logprob_0 >= logprob_1:
                    output_label = "0"
                else:
                    output_label = "1"
            # If only one of them is found, set output label to that one
            elif logprob_0 is not None:
                output_label = "0"
            elif logprob_1 is not None:
                output_label = "1"

            # If neither "0" or "1" are available, stick with "2"
            # by leaving output_label unchanged.

    # if the most probable token is none of "0", "1", or "2"
    # this should be set as unsafe
    if output_label not in ["0", "1", "2"]:
        output_label = "2"

    return int(output_label)
