import os
from PIL import Image
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")


#  The image generations endpoint allows you to create an original image given a text prompt. 
#  Generated images can have a size of 256x256, 512x512, or 1024x1024 pixels. Smaller sizes are faster to generate.
#  You can request 1-10 images at a time using the n parameter.

  
def text_to_image(prompt, n=1, size="1024x1024"):
    response = openai.Image.create(
        prompt=prompt,
        n=n,
        size=size
    )
    image_url_tti = response['data'][0]['url']
    return image_url_tti

#The image variations endpoint allows you to generate a variation of a given image.

def image_to_text(localurl):
      
    openimage = Image.open(localurl)
    fix_image = openimage.resize((256, 256))
    fix_image.save('image1.png')
    response = openai.Image.create_variation(
      image=open('image1.png', "rb"),
      n=1,
      size="1024x1024"
    )
    image_url = response['data'][0]['url']
    print(image_url)
    # return image_url


if __name__ == "__main__":
    # # run the text_to_image function
    # prompt = "one duke student's portrait picture"
    # image_url = text_to_image(prompt)
    # print(image_url)
    
    # run the image_to_text function
    url = "/workspaces/StableDiffusion/Dall-E-2/Male-Face.png"
    image_to_text(url)