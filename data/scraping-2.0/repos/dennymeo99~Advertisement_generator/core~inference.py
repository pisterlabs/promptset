import re
from io import BytesIO

import PIL.Image
import openai
import requests
import torch
from diffusers import DDIMScheduler, DiffusionPipeline
from torch import autocast
from transformers import AutoTokenizer

with open(f'../openai_key.txt', 'r') as file:
    openai.api_key = file.readline()

def generate_slogans(company_name, field, SEO_opt):
    base_model = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_auth_token="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")
    model = (torch.load("../models/text_model_torch.bin", map_location=torch.device('cuda')
    if torch.cuda.is_available() else torch.device('cpu')))

    if (SEO_opt == " "):
        input_text = f"What could it be a good advertising slogan for a company called {company_name} which operates in the {field} field?"
    else:
        input_text = f"What could it be a good advertising slogan for a company called {company_name} which operates in the {field} field, using the words '{SEO_opt}' ?"

    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    input_ids = input_ids.to(device)
    model_output = tokenizer.decode((model.generate(input_ids))[0])
    model_output = re.sub(r'<.*?>', '', model_output)

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": input_text}
        ]
    )
    gpt_output = re.sub("\"", "", completion.choices[0].message.content)

    return model_output, gpt_output


def generate_banner(company_name, field):
    prompt = f"generate an advertisement banner for a company called {company_name} which operates in the {field} field"
    model_image_path = "../generated_banners/Nvidia_CUDA_Logo.jpg"
    company_name = re.sub(" ", "_", company_name)

    # stable diffusion pipeline is available only using cuda
    if torch.cuda.is_available():
        pipe = DiffusionPipeline.from_pretrained("mimmodong/stable-diffusion-business", safety_checker=None,
                                                 torch_dtype=torch.float16,
                                                 use_auth_token="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").to("cuda")
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.enable_xformers_memory_efficient_attention()
        g_cuda = None
        g_cuda = torch.Generator(device='cuda')
        seed = 52362
        g_cuda.manual_seed(seed)

        negative_prompt = ""
        num_samples = 4
        guidance_scale = 7.5
        num_inference_steps = 24
        height = 512
        width = 512

        with autocast("cuda"), torch.inference_mode():
            images = pipe(
                prompt,
                height=height,
                width=width,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_samples,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=g_cuda
            ).images
        model_image = images[0]
        final_image = PIL.Image.fromarray(model_image)
        model_image_path = f"../generated_banners/{company_name}_model.jpg"
        final_image.save(model_image_path)



    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="1024x1024"
    )
    image_url = response['data'][0]['url']
    image_response = requests.get(image_url)
    image_data = image_response.content

    # Create a PIL Image object
    openai_image = PIL.Image.open(BytesIO(image_data))

    openai_image_path = f"../generated_banners/{company_name}_openai.jpg"
    openai_image.save(openai_image_path)


    return model_image_path, openai_image_path


if __name__ == '__main__':
    image = generate_banner("Mole cola", "beverage")
