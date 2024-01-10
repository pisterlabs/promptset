import logging
import tempfile
from langchain.chat_models import ChatOpenAI
from app_openai import doc_to_text, token_counter
from log.app_log import log_function_entry_exit


@log_function_entry_exit
def check_gpt_4(api_key):
    """
    Check if the user has access to GPT-4.

    :param api_key: The user's OpenAI API key.

    :return: True if the user has access to GPT-4, False otherwise.
    """
    try:
        ChatOpenAI(openai_api_key=api_key, model_name='gpt-4').call_as_llm('Hi')
        return True
    except Exception as e:
        logging.error(f"Error checking GPT-4 access: {e}")
        return False


@log_function_entry_exit
def check_gpt_3_5(api_key):
    """
    Check if the user has access to GPT-4.

    :param api_key: The user's OpenAI API key.

    :return: True if the user has access to GPT-4, False otherwise.
    """
    try:
        ChatOpenAI(openai_api_key=api_key, model_name='gpt-3.5-turbo').call_as_llm('Hi')
        return True
    except Exception as e:
        logging.error(f"Error checking GPT-3.5 access: {e}")
        return False


@log_function_entry_exit
def token_limit(doc, maximum=200000):
    """
    Check if a document has more tokens than a specified maximum.

    :param doc: The langchain Document object to check.

    :param maximum: The maximum number of tokens allowed.

    :return: True if the document has less than the maximum number of tokens, False otherwise.
    """
    text = doc_to_text(doc)
    count = token_counter(text)
    if count > maximum:
        logging.error(f"Token limit exceeded. Token count: {count}")
        return False
    return True


@log_function_entry_exit
def token_minimum(doc, minimum=2000):
    """
    Check if a document has more tokens than a specified minimum.

    :param doc: The langchain Document object to check.

    :param minimum: The minimum number of tokens allowed.

    :return: True if the document has more than the minimum number of tokens, False otherwise.
    """
    text = doc_to_text(doc)
    count = token_counter(text)
    if count < minimum:
        return False
    return True


@log_function_entry_exit
def check_key_validity(api_key):
    """
    Check if an OpenAI API key is valid.

    :param api_key: The OpenAI API key to check.

    :return: True if the API key is valid, False otherwise.
    """
    try:
        ChatOpenAI(openai_api_key=api_key).call_as_llm('Hi')
        print('API Key is valid')
        return True
    except Exception as e:
        print('API key is invalid or OpenAI is having issues.')
        print(e)
        return False


@log_function_entry_exit
def create_temp_file(github_diffs):
    """
    Create a temporary file from an uploaded file.

    :param github_diffs: The GitHub diffs to write to the temporary file.

    :return: The path to the temporary file.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='a') as temp_file:
        for diff_part in github_diffs:
            if diff_part is not None:
                temp_file.write(diff_part)
    return temp_file.name


@log_function_entry_exit
def create_chat_model(api_key, use_gpt_4):
    """
    Create a chat model ensuring that the token limit of the overall summary is not exceeded - GPT-4 has a higher token limit.

    :param api_key: The OpenAI API key to use for the chat model.

    :param use_gpt_4: Whether to use GPT-4 or not.

    :return: A chat model.
    """
    if use_gpt_4:
        return ChatOpenAI(openai_api_key=api_key, temperature=0, max_tokens=500, model_name='gpt-4')
    else:
        return ChatOpenAI(openai_api_key=api_key, temperature=0, max_tokens=250, model_name='gpt-3.5-turbo')
