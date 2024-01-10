import pandas as pd
from openai.embeddings_utils import get_embedding, cosine_similarity
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

df = pd.read_csv('./outputs/combined_data_first_200_rows.csv', encoding='latin-1')
prompts = list(df['product'])
images = []

model_id = "dreamlike-art/dreamlike-photoreal-2.0"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# version 1?
for i, prompt in enumerate(prompts):
    image = pipe(prompt).images[0]
    image.save(f'result_{i}.jpg')
    images.append(image)

# version 2?
def prompt_to_img(self, 
                      prompts, 
                      height=512, width=512, 
                      num_inference_steps=50, 
                      guidance_scale=7.5, 
                      img_latents=None):
        # convert prompt to a list
        if isinstance(prompts, str):
            prompts = [prompts]
        # get prompt embeddings
        text_embeds = self.get_prompt_embeds(prompts)
        # get image embeddings
        img_latents = self.get_img_latents(text_embeds,
                                      height, width,
                                      num_inference_steps,
                                      guidance_scale, 
                                      img_latents)
        # decode the image embeddings
        imgs = self.decode_img_latents(img_latents)
        # convert decoded image to suitable PIL Image format
        imgs = self.transform_imgs(imgs)
        return imgs

images = prompt_to_img(prompts)