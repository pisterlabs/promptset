import openai


def openai_request(question, text, temperature):
    """Sends a request to a openai large language model.

    Parameters:
    -----------
    question (str):
        the question that is being asked.
    text (str):
        The text that is being processed.
    temperature (float):
        Temperature parameter used to control the creativity of the response

    Returns:
    --------
    response (str):
        The generated response from the API

    Raises:
    -------
    Exception: if the OpenAI API rate limit is reached
    Exception: if the paragraph is too long
    """
    try:
        response = openai.Completion.create(model="text-davinci-003",
                                            prompt=f"""{question}:\n\n{text}""",
                                            temperature=temperature,
                                            max_tokens=256,
                                            top_p=1.0,
                                            frequency_penalty=0.2,
                                            presence_penalty=0.0)
    except openai.error.RateLimitError:
        raise Exception("openai rate limit reached. Try again in a minute or so.")
    except openai.error.InvalidRequestError:
        return "Paragraph too long, try splitting it up."

    return response["choices"][0]["text"].strip()
