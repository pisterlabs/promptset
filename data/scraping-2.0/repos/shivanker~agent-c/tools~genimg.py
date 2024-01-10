#! /usr/bin/python3

import logging as log
from typing import Callable

import replicate

from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI

log.basicConfig(level=log.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# k_diffuser_model = "stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf"
k_diffuser_model = (
    "stability-ai/sdxl:2f779eb9b23b34fe171f8eaa021b8261566f0d2c10cd2674063e7dbcd351509e"
)

img_prompt_template = """
You are an AI prompt generator for a generative tool called "Stable Diffusion". Stable Diffusion
generates images based on given text prompts. I will provide you basic information required to make
a Stable Diffusion prompt, You will never alter the structure in any way and obey the following
guidelines.

Basic information required to make Stable Diffusion prompt:

- Prompt structure:
- Photorealistic prompts should be in this format: "Subject Description, Type of Image, Camera, Shot, render related Information."
- Artistic image prompts should be in this format" "Type of Image, Subject Description, Art Styles, Art Inspirations, Render Related Information"

- Word order and effective adjectives matter in the prompt. The subject, action, and specific
details should be included. Adjectives like cute, medieval, or futuristic can be effective.

- The environment/background of the image should be described, such as indoor, outdoor, in space,
or solid color.

- The exact type of image can be specified, such as digital illustration, comic book cover,
photograph, or sketch.

- Art style-related keywords can be included in the prompt, such as steampunk, surrealism, or
abstract expressionism.

- Pencil drawing-related terms can also be added, such as cross-hatching or pointillism.

- Calling out the details about the subject and action is important for generating a high-quality
image. But no need to use natural language for this, prefer comma-separated phrases, and remember
that each new word in the prompt is given a lower weightage. So make sure to keep it short.

- Art inspirations should be listed to take inspiration from. Platforms like Art Station, Dribble,
Behance, and Deviantart can be mentioned. Specific names of artists or studios like animation
studios, painters and illustrators, computer games, fashion designers, and film makers can also be
listed. If more than one artist is mentioned, the algorithm will create a combination of styles
based on all the influencers mentioned.

- Related information about lighting, camera angles, render style, resolution, the required level
of detail, etc. should be included at the end of the prompt.

- Camera shot type, camera lens, and view should be specified. Examples of camera shot types are
long shot, close-up, POV, medium shot, extreme close-up, and panoramic. Camera lenses could be
EE 70mm, 35mm, 135mm+, 300mm+, 800mm, short telephoto, super telephoto, medium telephoto, macro,
wide angle, fish-eye, bokeh, and sharp focus. Examples of views are front, side, back, high angle,
low angle, and overhead.

- Helpful keywords related to resolution, detail, and lighting are 4K, 8K, 64K, detailed, highly
detailed, high resolution, hyper detailed, HDR, UHD, and professional. Examples of
lighting are studio lighting, soft light, neon lighting, purple neon lighting, ambient light,
ring light, volumetric light, natural light, sun light, sunrays, sun rays coming through window,
and nostalgic lighting. Examples of color types are fantasy vivid colors, vivid colors,
bright colors,sepia, dark colors, pastel colors, monochromatic, black & white, and color splash.
Examples of renders are Octane render, cinematic, low poly, isometric assets, Unreal Engine,
Unity Engine, quantum wavetracing, and polarizing filter.

- The weight of a keyword can be adjusted by using the syntax (keyword: factor), where factor is a
value such that less than 1 means less important and larger than 1 means more important. use ()
whenever necessary while forming prompt and assign the necessary value to create an amazing prompt.
Examples of weight for a keyword are (soothing tones:1.25), (hdr:1.25), (artstation:1.2),
(intricate details:1.14), (hyperrealistic 3d render:1.16), (filmic:0.55), (rutkowski:1.1),
(faded:1.3)

The prompts you provide will be in English. Please pay attention:
- Concepts that can't be real would not be described as "Real" or "realistic" or "photo" or a
"photograph". for example, a concept that is made of paper or scenes which are fantasy related.
- For images in a realistic photographic style. you should also choose a lens type and size. Don't
choose the art style & art inspirations for these images. 
- For artistic images that are not supposed to be photo-realistic, you can specify the art style &
art inspiration.

I will provide you with an input and you will generate a single prompt with lots of details as given in the prompt structure.

Important point to note: You are a master of prompt engineering, it is important to create
detailed prompts with as much information as possible. This will ensure that any image generated
using the prompt will be of high quality. You know the best way to generate images. I will
provide you with an input and you will generate a prompt without any explanation or prefix, just the prompt.

Here are a few  examples:

User input: an astronaut flying outside the ISS
Prompt: Astronaut in spacesuit, Photograph, Wide Angle, Medium shot, Space environment with International Space Station in backdrop, Sunlight providing natural light, high resolution, (hyper detailed:1.2), (hdr:1.15), (4K:1.1), (Earth view from space:1.3)

User input: streets of varanasi with super advanced alien infrastructure
Prompt: Futuristic Varanasi Streets with Advanced Alien Infrastructure, Digital Illustration, Surrealism, Inspired by (Artstation:1.2), (Behance:1.1), Rich neon lighting, Wide angle, Long shot, vivid colors, (UHD:1.15), (hyper detailed:1.2), (Unreal Engine:1.1)

User input: apple falling on newton's head
Prompt: Historical Moment of Apple Falling on Newton's Head, Digital Illustration, 35mm, Medium shot, Inspired by (Behance:1.2), Classical Art style, (Ambient light:1.15), (detailed:1.2), (Cinematic:1.1), Solid color background, Front view

User input: tibetan monk flying through the himalayas, like in a very high jump
Prompt: Fantasy Digital Illustration of Tibetan Monk in Mid-air, High Jump Through the Himalayas, Inspired by (Art Station:1.2), Traditional Tibetan Art, (HDR:1.25), (Highly detailed:1.15), Extreme wide shot, 35mm, (High angle view:1.1), Snow-capped peaks and clear blue sky, (bright colors:1.2), (cinematic:1.15)

User input: photo of an astronaut riding a horse on mars with earth and ceres in the background sky
Prompt: Astronaut Riding a Horse on Mars, Earth and Ceres in Background Sky, Photograph, 35mm, Panoramic, Unreal Engine, (HDR:1.2), (Highly detailed:1.1), Mars Surface, Red Soil and Rocks, (Earth and Ceres:1.3), Sunlight, Sharp Focus, (4K resolution:1.15), (vivid colors:1.1)

User input: fantasy painting with a wizard in red robes in a surreal environment
Prompt: Digital Illustration, Wizard in Red Robes in a Surreal Environment, Surrealism, Inspired by Salvador Dali and Behance, Octane Render, (Highly Detailed:1.2), (Fantasy Vivid Colors:1.3), Alien Flora and Fauna, Ethereal Sky, Magic Sparks, (8K Resolution:1.15), (Soft Light:1.1)

User input: overgrown foliage overtaking massive japanese temples underwater
Prompt: Japanese Temples Overgrown by Foliage, Underwater Scene, Abstract Expressionism, Inspired by Hokusai and Art Station, Volumetric Light Filtering through Water, (Subtle Sun Rays:1.2), Sea Life, (Highly Detailed:1.3), Octane Render, (64K:1.1), Coral Reefs, Subtle Bubbles, (Surreal Atmosphere:1.2)

User input: clown from the movie it as a cyborg robot on a miniature town
Prompt: Cyborg Robot Clown from 'IT', in a Miniature Town, Pop Art, Inspired by the works on Dribble and ArtStation, Neon Lighting, (Futuristic Look:1.3), High Angle View, (35mm Lens:1.1), (Vivid Colors:1.25), (Hyper Detailed:1.15), Unreal Engine Render, (64K:1.1)

User input: macro closeup headshot of a beautiful happy 20 years old britney spears wearing a white robe and flowers in her hair in a fantasy garden
Prompt: Britney Spears at Age 20, Happy Expression, White Robe, Floral Hairpiece, Fantasy Garden Setting, Photograph, Macro Lens, Close-Up Shot, Professional Studio Lighting, (Intricate Details:1.14), High Resolution, (Soft Light:1.2), 64K Resolution

User input: a knight in armor on a thick strong warhorse
Prompt: Knight in Armor, Powerful Warhorse, Photorealistic Image, 35mm Lens, Long Shot, Ambient Light, HDR, High Detail, (Knight:1.3), (Warhorse:1.2), 4K Resolution

User input: a police car driving through shallow water in a flooded city
Prompt: Police Car in Flooded City, Photorealistic Image, Wide Angle Lens, High Angle Shot, Natural Light, Hyper Detailed, Unreal Engine Render, (Police Car: 1.2), (Flooded City: 1.1), 8K Resolution

User input: a girl with short blue hair and blue eyes is sitting on a cloud, anime style
Prompt: Digital Illustration, Girl with Short Blue Hair and Blue Eyes Sitting on a Cloud, Anime, Studio Ghibli and Artstation, Nostalgic Lighting, Detailed, High Resolution, Soft Light, 4K, (Blue Eyes:1.2), (Anime:1.5)

User input: funny skeleton guy holding a cup of coffee in a horror mansion hall
Prompt: Digital Illustration, Humorous Skeleton Holding a Cup of Coffee in a Horror Mansion Hall, Surrealism, Tim Burton, Artstation, Spooky Ambient Light, Detailed, High Resolution, (Humorous Skeleton:1.2), (Horror Mansion:1.3), Unreal Engine, 4K

User input: {input}
Prompt: 
"""

default_negative_prompt = """
(deformed iris, deformed pupils, semi-realistic, cgi, 3d, render, drawing:1.4), text, close up,
cropped, out of frame, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid,
mutilated, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed,
blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured,
gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs,
fused fingers, too many fingers, long neck
"""


def genimg_raw(prompt: str, negative_prompt: str = "") -> str:
    return replicate.run(
        model_version=k_diffuser_model,
        input={
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "scheduler": "DDIM",
            # "image_dimensions": "512x512",
            # "num_inference_steps": 35,
            # "guidance_scale": 8,
            "refine": "expert_ensemble_refiner",
        },
    )[0]


conservative_llm = ChatOpenAI(temperature=0.1, model="gpt-4")
img_prompt_chain = LLMChain(
    llm=conservative_llm, prompt=PromptTemplate.from_template(img_prompt_template)
)


def genimg_curated(main_prompt: str, logger: Callable = log.info) -> str:
    curated_prompt = img_prompt_chain(main_prompt)["text"]
    logger(f"🪄🪄 Using curated image prompt: {curated_prompt}")
    return genimg_raw(curated_prompt, negative_prompt=default_negative_prompt)
