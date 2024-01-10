import openai
from settings import getSettings

settings = getSettings()


async def aInfer(text: str):
    """
    aInfer(text: str)

    Parameters:

    text (str): Text to generate completion for.

    Returns:

    content (str): Generated completion based on input text.
    model (str): Model used to generate completion.

    """

    response = await openai.Completion.acreate(
        model="text-curie-001",
        prompt=text,
        temperature=0,
        max_tokens=250,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    content: str = response.choices[0].text  # type: ignore
    model: str = f'gpt3-{response.model}'  # type: ignore
    return content, model
