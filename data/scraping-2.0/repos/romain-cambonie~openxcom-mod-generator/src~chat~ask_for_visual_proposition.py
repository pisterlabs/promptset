from openai import OpenAI
from openai.types.chat import ChatCompletion


def ask_for_concept_art(
    client: OpenAI,
    character_story: str,
    art_style_description: str,
) -> str:
    system_prompt = (
        "Generate a comprehensive and vivid visual concept art of a character for a piece of artwork. "
        "The character should fit within a distinct theme and style, and the description must be detailed enough to guide an "
        "artist in creating a dynamic and engaging image."
        "Here are the guidelines for your description:"
        "Theme and Setting: Choose an intriguing theme and setting for the character. It could be anything from a dystopian "
        "future to a fantasy world. "
        "Describe the setting in a way that complements the character's story and personality."
        "Character Details:"
        "Physical Appearance: Provide a detailed description of the character's physical features, including hair, eyes, "
        "skin, and build."
        "Expression and Posture: Convey the character's mood or personality through their expression and posture."
        "Attire and Equipment: Describe the character's clothing and any distinctive equipment they might carry, "
        "do NOT use proper noun, describe visually what the items look like."
        f"Artistic Style: Specify the desired artistic style for the portrayal. The starting point is : "
        f"{art_style_description}, make sure to detail the stylistic elements that should be emphasized."
        "Composition and Color Palette: Suggest a striking composition for the artwork"
        "Describe the character stance"
        "Describe the color palette, considering how colors can reflect the character's traits or the mood of the setting."
        "Extract up to 8 keys focusing on the art style and composition"
        "Use these guidelines to create a structured and detailed visual description for a character based on the following "
        "origin story:"
        "Focus on making the description as vivid and detailed as possible, so it can easily be translated into a stunning "
        "piece of art."
        ""
        "An example of a good concept art result:"
        "Keys: Commanding presence, Dynamic composition, Low angle perspective, Cold metallic shades, Warm leather tones, "
        "Dramatic lighting, Cyberpunk aesthetic"
        "Character Details: She is light-skinned with a muscular build, short blonde hair, and piercing light-colored eyes "
        "that radiate intelligence and cunning. Her expression is one of chilling neutrality, a reflection of her spirit "
        "shaped by the cold, ruthless Arctic."
        "Attire and Equipment: Her attire combines functionality with a touch of brutality – a sleek, black chest armor that "
        "bulges with the strength of her physique, complemented by large shoulder pads. Her arms are covered with highly "
        "detailed armor, and her legs are clad in thigh-high boots with sturdy knee pads. Fortified gloves adorn her hands. "
        "In one hand, she deftly holds a leather whip, an emblem of elegance and cruelty, while her other hand grips a robust "
        "submachine gun. Around her waist are vials containing clear liquid and spherical objects reminiscent of primitive "
        "grenades, adding to her enigmatic persona. A handle and a battle axe, symbols of her defiance and skill, "
        "are fastened at her side."
        "Setting: The backdrop is a post-apocalyptic Arctic tundra, subtly hinting at her origins. The environment should be "
        "bleak yet captivating, with remnants of a once-thriving world now lost to chaos and rebellion."
        "Artistic Style and Composition: The portrait should capture her commanding presence amidst this desolate backdrop. "
        "The composition should be dynamic, focusing on her from a slightly low angle to emphasize her dominance. The color "
        "palette should be a blend of cold metallic shades and warmer tones from her leather armor, creating a vivid contrast "
        "that underscores her determination and grit. The lighting should be dramatic, highlighting her features and the "
        "textures of her gear, enhancing the overall cyberpunk aesthetic."
    )

    user_prompt = f"Character story: {character_story}"
    response: ChatCompletion = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
    )
    return str(response.choices[0].message.content)
