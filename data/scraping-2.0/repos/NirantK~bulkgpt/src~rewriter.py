import openai
import os

openai.api_key = os.environ.get("OPENAI_API_KEY")


def rewrite(sku_description: str) -> str:
    """
    Rewrites a description using GPT3.

    Args:
        description (str): _description_

    Returns:
        str: _description_
    """
    prompt = f"""This is a product description. Rewrite this in your own words to make it more casual. Ensure all information about what the product contains is included in final output. Add a call to action for the reader. Make it readable for website visitors by adding line breaks where needed. In the end, add some lines about the fact that buying this product will support the artist mentioned. Keep the same meaning:\n\n{sku_description}"""

    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.0,
        max_tokens=2500,
        frequency_penalty=0.3,
        presence_penalty=0.15,
    )
    return str(response.choices[0].text)
