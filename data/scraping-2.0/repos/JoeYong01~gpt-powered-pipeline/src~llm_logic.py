"""contains functions to make calls to openai"""
import logging
import openai
from tenacity import (
    retry,
    wait_exponential,
    retry_if_exception_type,
    stop_after_attempt
)

logger = logging.getLogger("llm_logic.py")

@retry(
    retry=retry_if_exception_type(
        (
            openai.APITimeoutError,
            openai.InternalServerError,
            openai.RateLimitError
        )
    ),
    wait=wait_exponential(min = 1, max = 60),
    stop=stop_after_attempt(5)
)
def transcribe_audio(
    client: str,
    model: str,
    audio_file
) -> str:
    """
    Transcribes audio to text using OpenAI.

    Args:
        client (str): OpenAI client object to pass
        model (str): model for transcription
        audio_file (_type_): audio file to transcribe

    Returns:
        str: returns a transcription of an audio file
    """
    logger.info("running function: transcribe_audio.")
    try:
        with open(audio_file, "rb") as file:
            logger.debug("transcribing audio file.")
            transcript = client.audio.transcriptions.create(
                model = model,
                file = file,
                response_format='text'
            )
            logger.debug("returning transcript.")
            return transcript
    except openai.OpenAIError as e:
        logger.exception("OpenAI Exception in transcribe_audio: %e", e)
        raise e
    except Exception as e:
        logger.exception("Exception in transcribe_audio: %e", e)
        raise e

@retry(
    retry=retry_if_exception_type(
        (
            openai.APITimeoutError,
            openai.InternalServerError,
            openai.RateLimitError
            )
        ),
    wait=wait_exponential(min = 1, max = 60),
    stop=stop_after_attempt(5)
)
def call_completions(
    client,
    model: str,
    prompt: str,
    temperature: int,
    input_text: str
) -> bool:
    """
    Evaluates whether the prompt against the input text & returns a boolean value

    Args:
        client (_type_): OpenAI client object to be passed
        model (str): model for completions
        prompt (str): prompt for desired return value
        temperature (int): lower values (0-0.2) are preferred to fulfil prompt conditions
        input_text (str): transcription to be evaluated

    Returns:
        bool: 1 / 0
    """
    logger.info("running function: call_completions.")
    try:
        logger.debug("evaluating transcription.")
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": input_text}
            ],
            temperature=temperature
        )
        logger.debug("returning boolean response.")
        return response.choices[0].message.content
    except openai.OpenAIError as e:
        logger.exception("OpenAI exception in call_completions: %s", e)
        raise e
    except Exception as e:
        logger.exception("Exception in call_completions: %s", e)
        raise e
