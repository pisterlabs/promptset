from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize the OpenAI client
try:
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY_JF_2"),
    )
    print("OpenAI client successfully initialized.")

    # Attempt to create a new image
    print("Generating image...")
    response = client.images.generate(
        model="dall-e-3",
        prompt="""In this complex and imaginative battlefield, the contrasts of war and whimsy collide. The scene is set in a rugged, war-torn landscape, reminiscent of intense conflict zones. The ground is marked with the scars of battle; craters from explosions and debris scattered chaotically. Amidst this devastation, a surreal twist unfolds.

At the forefront, a cat commandeers a tank, but this isn't just any tank – it's brightly colored, almost cartoonish, lending a playful air to its otherwise formidable presence. The cat, wearing a comical general's hat, peers out with a look of fierce determination and a hint of mischief.

The surrounding battlefield is a dizzying mix of intense violence and absurdity. Real soldiers engage in battle, their expressions a mix of focus and bewilderment at the bizarre elements around them. These moments of stark realism – the grit, the determination, the sweat, and blood – contrast sharply with the surreal aspects.

Toy soldiers, styled like classic playthings, are interspersed among the real troops. Some are positioned in dramatic falls, adding a layer of dark humor to the scene. In the background, exaggerated, almost cartoonish explosions light up the sky in a vivid display of colors, their loud booms juxtaposed with the comical 'pew-pew' sounds of toy guns.

The landscape is littered with both realistic and fantastical elements. Realistic military vehicles are parked next to oversized, playful versions, blurring the lines between a serious warzone and a child's imaginative play area. Animals in uniform perform over-the-top action maneuvers, adding to the chaos and whimsy.

Blood and dirt mix with confetti and paint, illustrating the surreal merger of harsh warfare and playful fantasy. The scene is a visual representation of intense violence softened by elements of humor and absurdity, creating a paradoxical world where the horrors of war meet the innocence of play.

Double the carnage and death by 24.

""",
        size="1024x1024",
        quality="hd",
        n=1,
    )

    # Extract and print the image URL
    image_url = response.data[0].url
    print(f"Image successfully generated: {image_url}")

except Exception as e:
    # Print any errors that occur
    print(f"An error occurred: {e}")
