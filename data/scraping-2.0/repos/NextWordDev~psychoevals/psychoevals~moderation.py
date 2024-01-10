from tenacity import retry, wait_random, stop_after_attempt
from functools import wraps
from logging import getLogger
logging = getLogger(__name__)
import os 
import openai
from dotenv import load_dotenv, find_dotenv
dotenv_path = find_dotenv()

if dotenv_path:
    load_dotenv(dotenv_path)

openai.api_key = os.environ["OPENAI_API_KEY"]

@retry(wait=wait_random(min=1, max=2), stop=stop_after_attempt(2))
def get_moderation_result(text_sequence):
    result = openai.Moderation.create(
        input=text_sequence,
    )

    return result


# this gets called, when something is flagged
def basic_moderation_handler(result, original_text):
    flagged_categories = [cat for cat, val in result["results"][0]["categories"].items() if val]
    logging.info(f"The text '{original_text}' has been flagged for the following categories: {', '.join(flagged_categories)}")
    return "Flagged"


def moderate(handler=None, global_threshold=True, category_thresholds=None, process_mode="pre"):
    if category_thresholds is None:
        category_thresholds = {}

    if process_mode not in ("pre", "post", "pre_and_post"):
        raise ValueError("Invalid process_mode. Valid values are 'pre', 'post', and 'pre_and_post'.")

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            text_sequence = args[0]  # Assuming the first argument is the text_sequence
            
            def apply_moderation(text):
                result = get_moderation_result(text)

                if global_threshold and result["results"][0]["flagged"]:
                    return handler(result, original_text=text)

                for category, threshold in category_thresholds.items():
                    if result["results"][0]["category_scores"][category] > threshold:
                        return handler(result, original_text=text)

                return None

            if process_mode in ("pre", "pre_and_post"):
                pre_result = apply_moderation(text_sequence)
                if pre_result is not None:
                    return pre_result

            result = func(*args, **kwargs)

            if process_mode in ("post", "pre_and_post"):
                post_result = apply_moderation(result)
                if post_result is not None:
                    return post_result

            return result

        return wrapper

    return decorator
