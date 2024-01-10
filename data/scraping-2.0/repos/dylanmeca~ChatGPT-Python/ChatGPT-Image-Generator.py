import openai
import gradio as gr
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
import torch

model_id = "stabilityai/stable-diffusion-2"

# Use the Euler scheduler here instead
scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, revision="fp16", torch_dtype=torch.float16)
pipe = pipe.to("cuda")

# Getting responses using the OpenAI API
def response_chatgpt(api_key, prompt):
  # OPENAI API KEY
  openai.api_key = api_key
  prompt = (f"I want you to analyze this prompt that is used to generate an image based on it and with a similar structure, I want you to write a prompt but so that it generates images of {prompt}")
  response = openai.Completion.create(
      engine="text-davinci-003",
      prompt=prompt,
      max_tokens=1024
  )
  # Displaying the answer on the screen
  result = response["choices"][0]["text"]
  # Generating image
  image = pipe(result, height=768, width=768, guidance_scale = 10).images[0]
  image.save("sd_image.png")
  return result, image
      
# User input
chatbot = gr.Interface(
    fn=response_chatgpt, 
    inputs=["text", "text"],
    outputs=["text", "image"]
)
chatbot.launch()   
