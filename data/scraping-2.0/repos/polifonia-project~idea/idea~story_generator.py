import ast
import openai

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)  # for exponential backoff


def response_parser(response):
    try:
        response = ast.literal_eval(response)
    except (ValueError, TypeError, SyntaxError):
        response = ""
    return response


@retry(
    retry=retry_if_exception_type((openai.error.APIError,
                                   openai.error.APIConnectionError,
                                   openai.error.RateLimitError,
                                   openai.error.ServiceUnavailableError,
                                   openai.error.Timeout)),
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(10)
)
def story_generator(input_data: dict):
    """

    Parameters
    ----------
    input_data : input data formatted as dict/JSON, including
                 - persona
                 - specifications
                 - TODO: existing ontology

    Returns
    -------
    story : story text
    """
    # 1st interation
    # TODO: refine the prompt template
    prompt_1 = "Given persona: {}, and specifications: {}".format(input_data['persona'], input_data['specifications'])
    conversation_history = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_1}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation_history
    )
    response = response.choices[0].message.content
    print("Output is: \"{}\"".format(response))

    # 2nd iteration
    # TODO: refine the prompt template
    prompt_2 = ""
    conversation_history.append({"role": "assistant", "content": response})  # previous response
    conversation_history.append({"role": "user", "content": prompt_2})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation_history
    )
    response = response.choices[0].message.content
    print("Output is: \"{}\"".format(response))

    # TODO: add more iterations if need
    story = response_parser(response)

    return story
