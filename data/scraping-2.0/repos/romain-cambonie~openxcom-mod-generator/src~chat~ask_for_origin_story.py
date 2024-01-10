from openai import OpenAI
from openai.types.chat import ChatCompletion


def ask_for_origin_story(
    client: OpenAI,
    character_name: str,
    equipment_description: str,
    appearance_description: str,
) -> str:
    system_prompt = (
        "You are tasked with creating a short origin story for a fictional character. "
        "You will receive three key pieces of information: (1) the character's name, "
        "(2) a YAML payload detailing the character's equipment, and "
        "(3) an image that shows some characteristics of the character's appearance. "
        "Your job is to weave these elements together into a compelling and imaginative origin story. "
        "The story should be concise, no more than a few paragraphs, and should creatively incorporate specific details from "
        "the YAML payload and the visual cues from the image. "
        "The tone and style of the story should align with the genre suggested by the character's name and appearance. "
        "Be imaginative and ensure that the equipment and visual traits play a significant role in the character's background "
        "and the events that shaped them."
        "Pay special attention to match all visual description details such as gender, race, skin color, hair color and so on "
    )

    user_prompt = (
        f"Character Name: {character_name}\n\nEquipment: {equipment_description}\n\nAppearance: "
        f"{appearance_description}\n\nBased on the above details, create a short origin story for the character."
    )

    response: ChatCompletion = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
    )
    return str(response.choices[0].message.content)
