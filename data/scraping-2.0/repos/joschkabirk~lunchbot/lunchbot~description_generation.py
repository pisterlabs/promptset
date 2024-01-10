"""Generate a description for a meal based on a prompt."""
import logging

from openai import OpenAI

logger = logging.getLogger(__name__)


def get_food_description(
    meal_name: str,
    return_prompt_answer: bool = False,
    system_content: str = None,
):
    """Generate a description for a meal based on a prompt.

    Parameters
    ----------
    meal_name : str
        The name of the meal to generate a description for.
    return_prompt_answer : bool, optional
        Whether to return the prompt and the answer, by default False
    system_content : str, optional
        The system content to use for the prompt, by default None

    Returns
    -------
    str or tuple
        The answer or a tuple of the prompt and the answer (if return_prompt_answer=True).
    """

    client = OpenAI()

    if system_content is None:
        system_content = (
            "You are a five star chef. You know how to describe food. "
            "But don't start always with 'Indulge in ...'"
        )

    prompt = meal_name + " - please describe this meal in two sentences."

    logger.info(f"Prompt: {prompt}")
    logger.info(f"System Content: {system_content}")

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": prompt},
        ],
    )

    response = completion.choices[0].message.content

    if return_prompt_answer:
        return prompt, response

    return response
