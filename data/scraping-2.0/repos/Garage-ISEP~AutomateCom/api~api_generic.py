import os
import textwrap
from PIL import Image, ImageDraw, ImageFont
import openai

openai.api_key = os.getenv("OPENAI_APIKEY")
PROMPT_PATH = "api/generic/description_prompt.txt"

def generate_text(title):
    """
    Generate a description for an event post using OpenAI API.

    :param title: Title of the event for the prompt
    :return: Generated description of the event.
    """
    prompt = str(open(PROMPT_PATH, "r")) + title
    response = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=1000, n=1, stop=None,
                                        temperature=0.5)
    generated_text = response.choices[0].text.strip()
    return generated_text

def generate_image(event_name, lab, description, date, hour, location):
    """
    Generate an image for an instagram post based on a template
    :param event_name: Title of the event
    :param lab: Name of the lab doing the event
    :param description: Description of the event
    :param date: Date of the event
    :param hour: Hour of the event
    :param location: Location of the event
    :return: image with all this information.
    """
    # Create the title and description for the Instagram post
    title = f"{event_name}"
    lines = textwrap.wrap(description, width=30)
    description = "\n".join(lines)

    # Generate the image with the post content -- NEED UPDATE
    image_path = os.path.join("assets", f"{lab}.png")
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    title_font = ImageFont.truetype("ressources/fonts/LeagueSpartan-Bold.ttf", 50)
    desc_font = ImageFont.truetype("ressources/fonts/LeagueSpartan-Bold.ttf", 40)
    draw.text((150, 535), title, fill=(0, 0, 0), font=title_font)
    draw.text((110, 250), f"{hour}", fill=(255, 255, 255), font=desc_font)
    draw.text((830, 250), f"{date}", fill=(255, 255, 255), font=desc_font)
    draw.text((450, 950), f"{location}", fill=(255, 255, 255), font=desc_font)

    # Save the generated image
    post_image_path = "output/generated_post.png"
    img.save(post_image_path)
    print("Instagram post generated!")

    # Display the image
    img.show()


if __name__ == '__main__':
    pass
