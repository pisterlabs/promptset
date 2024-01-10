# Copyright (c) 2023 ANSYS, Inc. All rights reserved
"""Module for exceptions."""


class ValidationErrorException(Exception):
    """Exception class for LLM format errors.

    Parameters
    ----------
    message : str, optional
        Message to be raised, by default "Output message from LLM is not properly formed.".

    llm_output : str, optional
        Raw output from the LLM, by default ``None``.
    """

    def __init__(
        self,
        message: str = "Output message from LLM is not properly formed.",
        llm_output: str = None,
    ):
        """Exception initialization."""
        self.message = message
        if llm_output is not None:
            message = message + "\n LLM raw output: \n\n" + llm_output
        super().__init__(self.message)


class EmptyOpenAIResponseException(Exception):
    """Exception class for OpenAI empty responses.

    Parameters
    ----------
    message : str, optional
        Message to be raised, by default "The response from OpenAI is empty.".
    """

    def __init__(self, message: str = "The response from OpenAI is empty."):
        """Exception initialization."""
        self.message = message
        super().__init__(self.message)
