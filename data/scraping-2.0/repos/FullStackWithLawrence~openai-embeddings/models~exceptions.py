# -*- coding: utf-8 -*-
"""Module exceptions.py"""

import openai


# pylint: disable=too-few-public-methods
class OpenAIResponseCodes:
    """Http response codes from openai API"""

    HTTP_RESPONSE_OK = 200
    HTTP_RESPONSE_BAD_REQUEST = 400
    HTTP_RESPONSE_INTERNAL_SERVER_ERROR = 500


class ModelConfigurationError(Exception):
    """Exception raised for errors in the configuration."""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class ModelValueError(Exception):
    """Exception raised for errors in the configuration."""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class ModelIlligalInvocationError(Exception):
    """Exception raised when the service is called by an unknown service."""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


EXCEPTION_MAP = {
    ModelValueError: (OpenAIResponseCodes.HTTP_RESPONSE_BAD_REQUEST, "BadRequest"),
    ModelConfigurationError: (OpenAIResponseCodes.HTTP_RESPONSE_INTERNAL_SERVER_ERROR, "InternalServerError"),
    ModelIlligalInvocationError: (OpenAIResponseCodes.HTTP_RESPONSE_INTERNAL_SERVER_ERROR, "InternalServerError"),
    openai.APIError: (OpenAIResponseCodes.HTTP_RESPONSE_BAD_REQUEST, "BadRequest"),
    ValueError: (OpenAIResponseCodes.HTTP_RESPONSE_BAD_REQUEST, "BadRequest"),
    TypeError: (OpenAIResponseCodes.HTTP_RESPONSE_BAD_REQUEST, "BadRequest"),
    NotImplementedError: (OpenAIResponseCodes.HTTP_RESPONSE_BAD_REQUEST, "BadRequest"),
    openai.OpenAIError: (OpenAIResponseCodes.HTTP_RESPONSE_INTERNAL_SERVER_ERROR, "InternalServerError"),
    Exception: (OpenAIResponseCodes.HTTP_RESPONSE_INTERNAL_SERVER_ERROR, "InternalServerError"),
}


class ConfigurationError(Exception):
    """Exception raised for errors in the configuration."""

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)
