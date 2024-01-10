import openai
import os
from gptprobe.utils.enginedefaults import DEFAULT_ENGINE, DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS
from gptprobe.keysinenviron import keys_in_environ 

# Main wrapper function for calling ChatGPT using a choice of key stored in environ, and params stored in environ


def ask_for_text_from_question(question,
                               key_choice:int=0,
                               engine=None,
                               max_tokens=None,
                               temperature=None,
                               as_dict=False):
    """ Use open ai keys in environment variables like OPEN_AI_KEY_0 to ask a question
    :param key_choice  0, 1, or 2  typically
    :return:
    """
    assert keys_in_environ(),' See https://github.com/microprediction/gptprobe/ README for instructions on inserting open ai keys into environment variables' 
    
    if engine is None:
        engine = os.environ.get('OPEN_AI_ENGINE') or DEFAULT_ENGINE

    if temperature is None:
        try:
            temperature = float(os.environ.get('OPEN_AI_TEMPERATURE'))
        except TypeError:
            temperature = DEFAULT_TEMPERATURE

    if max_tokens is None:
        try:
            max_tokens = int(os.environ.get('OPEN_AI_MAX_TOKENS'))
        except TypeError:
            max_tokens = DEFAULT_MAX_TOKENS


    # Creds
    api_env = f"OPEN_AI_KEY_{key_choice}"
    api_key = os.environ.get(api_env)
    if api_key is None:
        raise ValueError(f'The environment variable {api_env} must be set)')
    openai.api_key = api_key

    response = openai.Completion.create(
        engine=engine,
        prompt=question,
        max_tokens=max_tokens,
        stop=None,
        temperature=temperature
    )
    response_text = response.choices[0].text.strip()
    return {'success':1,'response':response_text} if as_dict else response_text


ask_for_text = ask_for_text_from_question


if __name__=='__main__':
    print(ask_for_text_from_question(question="Is this a good question?"))
