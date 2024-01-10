from django.conf import settings
from ..models import Word
import logging
import openai

logger = logging.getLogger(__name__)
client = openai.OpenAI()

def get_language_specific_instructions():
    language = settings.TARGET_LANGUAGE

    if language == 'Japanese':
        instructions = ' with furigana'
    else:
        instructions = ''

    return instructions


def send_prompt(manual_mode=False, simulate_error=None):
    language = settings.TARGET_LANGUAGE
    language_specific_instructions = get_language_specific_instructions()

    try:
        # simulate errors if in manual mode
        if manual_mode:
            if simulate_error == 'APIError':
                raise ValueError(
                    "OpenAI API returned an API Error: simulated error")
            elif simulate_error == 'APIConnectionError':
                raise ValueError(
                    "Failed to connect to OpenAI API: simulated error")
            elif simulate_error == 'RateLimitError':
                raise ValueError(
                    "OpenAI API request exceeded rate limit: simulated error")

            logger.info("Manual mode: Returning simulated response")
            return {"choices": [{"message": {"content": "Sample response for testing."}}]}

        content_message = f"Request:\n"
        content_message += f"Word in {language}: Give me a useful word to know"
        word_objects = Word.objects.filter(language=language)
        word_list = [word.word for word in word_objects]
        if word_list:
            word_string = ', '.join(word_list)
            content_message += f" that is not in this list: [{word_string}]"
        content_message += f".\nDefinition in {language}: Give the definition of the word in {language}{language_specific_instructions}.\n"
        content_message += f"Context Sentence in {language}: Offer a sentence using the word in {language}{language_specific_instructions}.\n"
        content_message += f"\nResponse Structure:\nWord: [In {language}]\n"
        content_message += f"Definition: [In {language}{language_specific_instructions}]\n"
        content_message += f"Context Sentence: [In {language}{language_specific_instructions}]\n"
        content_message += f"Translation:\nWord in English:\n"
        content_message += f"Definition in English:\n"
        content_message += f"Context Sentence in English:"

        logger.info("Making API call to fetch word information")
        logger.info(content_message)

        response = client.chat.completions.create(
            messages=[
                {"role": "user", "content": content_message},
            ],
            model=settings.OPENAI_API_MODEL,
        )
        logger.info("API call successful")
        return response
    except openai.APIConnectionError as e:
        logger.error(f"Failed to connect to OpenAI API: {str(e)}")
        raise ValueError(f"Failed to connect to OpenAI API: {str(e)}")
    except openai.RateLimitError as e:
        logger.error(f"OpenAI API request exceeded rate limit: {str(e)}")
        raise ValueError(f"OpenAI API request exceeded rate limit: {str(e)}")
    except openai.APIError as e:
        logger.error(f"OpenAI API returned an API Error: {str(e)}")
        raise ValueError(f"OpenAI API returned an API Error: {str(e)}")
