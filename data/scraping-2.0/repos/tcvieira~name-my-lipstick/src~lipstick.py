import streamlit as st
from openai import OpenAI
from PIL import Image
import requests
from io import BytesIO
from colors import hex2rgb, NAMES_ORIGINAL
from prompts import ZERO_SHOT_PROMPT_TEMPLATE, FEW_SHOT_PROMPT_TEMPLATE, MANY_SHOT_PROMPT_TEMPLATE, IMAGE_AD_PROMPT, IMAGE_DESCRIPTION_PROMPT
from utils import scroll_to_top

PROMPTS = {'zero-shot': ZERO_SHOT_PROMPT_TEMPLATE, 'few-shot': FEW_SHOT_PROMPT_TEMPLATE, 'many-shot': MANY_SHOT_PROMPT_TEMPLATE}

def llm_text(prompt: str):
    client = OpenAI()

    response = client.chat.completions.create(
        model=st.session_state['model'],
        temperature=st.session_state['temperature'],
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

def llm_image(prompt: str):
    client = OpenAI()

    quality = 'standard'

    if st.session_state['image-model'] == 'dall-e-3':
        quality = 'hd'

    response = client.images.generate(
            model=st.session_state['image-model'],
            prompt=prompt,
            size="1024x1024",
            quality=quality,
            n=1,
        )

    image_url = response.data[0].url
    return requests.get(image_url)


def generate_ad_image(name: str, color_rgb: str):
    st.components.v1.html(scroll_to_top)
    with st.status("Calling OpenAI API...", expanded=True) as status:

        image_description = IMAGE_AD_PROMPT.format(color=color_rgb, name=name)

        if  st.session_state['llm-image-prompt']:
            description_prompt = IMAGE_DESCRIPTION_PROMPT.format(color=color_rgb, name=name)

            status.update(label='Generating image description...', expanded=True)
            image_description = llm_text(description_prompt)
            print(image_description)

        status.update(label='Generating image...', expanded=True)

        response = llm_image(image_description)
        status.update(label=f'{name}', state='complete', expanded=True)
        img = Image.open(BytesIO(response.content))
        st.image(img, caption=f'Image Prompt: {image_description}')


def generate_name(similar_colors_rgb: dict):
    color = st.session_state['color']

    with st.status("Calling OpenAI API...", expanded=True) as status:
        for prompt_name in PROMPTS.keys():
            if prompt_name == 'few-shot':
                examples = []
                for idx, rgb_color in similar_colors_rgb.items():
                    template = f"<rgb>{rgb_color}<rgb>, <name>{NAMES_ORIGINAL[idx]}<name>"
                    examples.append(template)
                prompt = PROMPTS[prompt_name].format(color=hex2rgb(color), examples='\n'.join(examples))
            else:
                prompt = PROMPTS[prompt_name].format(color=hex2rgb(color))

            status.update(label=f"💄 Lipstick name generated with {prompt_name}!", expanded=True)
            generated_name = llm_text(prompt)
            st.session_state['generated_names'].append({'name': generated_name, 'prompt': prompt_name, 'color': color})
    status.update(label=f"💄 Lipstick name generation completed!", state="complete", expanded=True)


