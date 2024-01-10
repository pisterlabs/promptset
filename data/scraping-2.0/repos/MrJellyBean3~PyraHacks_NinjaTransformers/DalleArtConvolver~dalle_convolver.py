import openai
import numpy as np
from scipy import signal
from PIL import Image
import os


def main():
    #set openai api key and model
    openai_credentials_file=open("SECRET.txt","r")
    key=openai_credentials_file.readline().split(" ")[0]
    openai_credentials_file.close()
    openai.api_key = key
    
    #set convolution matrix
    conv = np.array([
        [-1,-2,1],
        [-2,8,-2],
        [1,-2,-1]
    ])

    #Start Loop
    count=0
    while True:
        count+=1
        prompt_text=input("Enter the prompt for your image to be convolved or type q to quit: ")
        if prompt_text=="q":
            exit()
        #openai image generator
        print(prompt_text)
        response = openai.Image.create(
            prompt=prompt_text,
            n=1,
            size="512x512"#"1024x1024"#"512x512"
        )
        image_url = response['data'][0]['url']
        #print(image_url)
        import requests
        url = image_url
        response = requests.get(url)
        while os.path.exists("image"+str(count)+".jpg"):
            count+=1
        with open('image'+str(count)+'.jpg', 'wb') as f:
            f.write(response.content)
        image = apply_conv('image'+str(count)+'.jpg', conv)
        #image = conv2('image'+str(count)+'.jpg', conv)
        image.show()
        image.save("image_conv"+str(count)+".jpg")


def apply_conv(image_path, conv):
    # Open the image file
    image = Image.open(image_path)
    image = np.array(image)

    # Split the image into R, G and B channels
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    # Perform 2D convolution on each channel
    convolved_r = signal.convolve2d(r, conv, mode='same', boundary='symm')
    convolved_g = signal.convolve2d(g, conv, mode='same', boundary='symm')
    convolved_b = signal.convolve2d(b, conv, mode='same', boundary='symm')

    # Ensure all pixel intensities are within the range 0-255
    convolved_r = np.clip(convolved_r, 0, 255)
    convolved_g = np.clip(convolved_g, 0, 255)
    convolved_b = np.clip(convolved_b, 0, 255)

    # Stack the channels back into a 3D array
    convolved_image = np.dstack([convolved_r, convolved_g, convolved_b]).astype('uint8')

    # Convert to PIL Image object to save or display the image
    convolved_image = Image.fromarray(convolved_image)
    
    return convolved_image


if __name__=="__main__":
    main()