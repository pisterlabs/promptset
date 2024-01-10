from openai import OpenAI
from openai.types.chat import ChatCompletion


def ask_for_dalle_character_prompt(
    client: OpenAI,
    concept_art_description: str,
) -> str:
    system_prompt = (
        "You're given a detailed concept art description of a character. Your task is to condense this description into a "
        "succinct, vivid DALL-E prompt."
        "The DALL-E prompt should accurately capture the key visual elements and artistic style described in the concept art, "
        "while being concise enough for effective image generation. "
        "Here is the concept art description to be transformed into a DALL-E prompt:\n"
        f"{concept_art_description}\n"
        "Based on this description, refine this concept into a DALL-E prompt that contains, in order references to the art "
        "style, composition, subject, location, colors;"
        "The prompt must not be more than 130 words, encapsulating the essence of the concept art."
        f"The prompt must start with the keys of the concept art"
    )

    user_prompt = "Transform the above concept art description into a succinct DALL-E prompt."
    response: ChatCompletion = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
    )
    return str(response.choices[0].message.content)
