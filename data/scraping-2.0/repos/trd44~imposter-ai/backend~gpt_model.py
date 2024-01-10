"""
gpt_model.py

Author: Christian Welling
Date: 12/3/2023
Company: ImposterAI
Contact: csw73@cornell.edu

Contains class for accessing GPTModels through the openai API. Must specify
what model id is to be used as well as additional parameters needed to
make a successful OpenAI API request.
"""

# region General/API Imports
import os
import openai

# endregion

# region Backend Imports
from backend.ai_model import AIModel
from backend.logger import LOGGER

# endregion


class GPTModel(AIModel):
    def __init__(self):
        """
        Access OpenAI GPT model with class.

        Presumes that the API key is set in the environment already...
        """
        self._model_id = "gpt-4"

    def set_model(self, model_id: str) -> None:
        """
        Sets the id of the model to be used.

        Arguments:
            model_id (str): The id of the model to be set.
        """
        # Set the model id
        self._model_id = model_id

    def make_request(self, conversation_messages):
        """
        Makes a request to the model.

        Args:
            conversation_messages (json): Input messages for the api call.

        Returns:
            JSON: The message response received from the "assistant".
                  Returns None if there's an error.
        """

        # Assigns the environment variable for the OPENAI API KEY to openai.api_key.
        # TODO: This should ideally be done once, perhaps during initialization.
        openai.api_key = os.getenv("OPENAI_API_KEY")

        ret = None
        # Make API request
        try:
            completion = openai.ChatCompletion.create(
                model=self._model_id,
                messages=conversation_messages,
                temperature=1.2,
            )
            ret = completion.choices[0].message

        # Error handling for different types of API errors, correctly categorized
        # Section 1: Service Errors
        except openai.error.APIConnectionError as e:
            LOGGER.error(f"Failed to connect to OpenAI API: {e}")
        except openai.error.ServiceUnavailableError as e:
            LOGGER.error(f"OpenAI API service is currently unavailable: {e}")
        except openai.error.RateLimitError as e:
            LOGGER.error(f"OpenAI API request exceeded rate limit: {e}")
        except (openai.error.Timeout, openai.error.TryAgain) as e:
            LOGGER.error(f"OpenAI API request timeout: {e}")

        # Section 2: Authentication Errors
        except openai.error.AuthenticationError as e:
            LOGGER.error(f"OpenAI API key cannot be authenticated: {e}")
        except openai.error.PermissionError as e:
            LOGGER.error(f"OpenAI API request does not have permission: {e}")
        except openai.error.SignatureVerificationError as e:
            LOGGER.error(f"OpenAI API signature cannot be verified: {e}")

        # Section 3: Request Errors
        except openai.error.InvalidAPIType as e:
            LOGGER.error(f"OpenAI API request contained invalid API Type: {e}")
        except openai.error.InvalidRequestError as e:
            LOGGER.error(f"OpenAI API request is invalid: {e}")

        # Section 4: General Errors
        except openai.error.APIError as e:
            # Handle general API error here, for instance, retry or log the error.
            LOGGER.error(f"OpenAI API returned an API Error: {e}")

        # Section 5: Unknown Errors
        except Exception as e:
            # Handle any other unknown errors
            LOGGER.error(f"An unknown error occurred: {e}")

        return ret
