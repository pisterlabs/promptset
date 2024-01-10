import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image
import shutil
import os
from PIL import Image
import random
from openai import OpenAI

client = OpenAI()

if os.path.exists("gen_imgs"):
    shutil.rmtree("gen_imgs")

os.makedirs("gen_imgs")

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe = pipe.to("cuda")
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)


url = "https://huggingface.co/datasets/patrickvonplaten/images/resolve/main/aa_xl/000000009.png"

init_image = load_image("test.webp").convert("RGB")
prompt = "a photo of an astronaut riding a horse on mars"
image = pipe(prompt, image=init_image,num_inference_steps=75,high_noise_frac=0.7).images[0]

action_map = [
    "S",
    "V",
    "AC",
    "X",
    "T",
    "W",
    "X",
    "F",
    "Z",
    "S",
    "TO",
    "X"
]
j = 0
initial_prompt = """
    You are a helpful sentence editing AI that is fluent writing prompts for stable diffusion models.
    We are writing prompts that will go into Stability's Stable Diffusion model.
    The prompts should be written in a way that is easy for the model to understand.
    Overall, the theme should be about space, the colors should be vibrant, and the tone should be psychedelic but towards the minimalist side.
    You will be provided with a prompt that you will edit based on an associated command.
    The commands are as follows:
    S: Change the subject of this sentence.
    V: Change the verb of this sentence.
    AC: Change the action of this sentence.
    X: Generate a new sentence.
    T: Change the topic of this sentence.
    W: Change the wording of this sentence.
    C: Change the camera angle of this sentence.
    X: Generate a new sentence.
    F: Change the focus of this sentence.
    Z: Change the tone of this sentence.
    S: Change the setting of this sentence.
    TO: Change the topic of this sentence.
    X: Generate a new sentence.    
    You will return a new sentence based on the command.
    """
prompt = "The image of a minimalist yet colorful astronaut journeying through a psychedelic space is ready for you to view. It combines the aesthetics of minimalism and psychedelia, creating a serene yet visually striking depiction of space."
for _ in range(1000):
    num = random.randint(0, len(action_map) - 1)
    action = action_map[num]
    change_command = f"{action}:" if action_map[num] != "X" else "Generate a new sentence."
    print(change_command + prompt)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {"role": "system", "content": initial_prompt},
            {"role": "user", "content": change_command + prompt},
        ]
    )


    prompt = response.choices[0].message.content

    print(prompt)

    # Original dimensions
    original_width, original_height = image.size

    # Define the new crop box (15 pixels inwards from each side)
    crop_box = (15, 15, original_width - 15, original_height - 15)

    # Crop the image
    cropped_image = image.crop(crop_box)

    # Resize the cropped image back to the original size
    image = cropped_image.resize((original_width, original_height))

    # Paste the cropped image onto this new image
    if j == 0:
        image = pipe(prompt, image=image).images[0]
    else:
        image = load_image(f"gen_imgs/test_{j-1}.png")
            # Original dimensions
    original_width, original_height = image.size

    # Define the new crop box (15 pixels inwards from each side)
    crop_box = (100, 150, original_width - 260, original_height - 100)

    # Crop the image
    cropped_image = image.crop(crop_box)

    # Resize the cropped image back to the original size
    image = cropped_image.resize((original_width, original_height))
    image = pipe(prompt, image=image, mask_image=image, strength=0.45).images[0]
    image.save(f"gen_imgs/test_{j}.png")
    j += 1


print(image)