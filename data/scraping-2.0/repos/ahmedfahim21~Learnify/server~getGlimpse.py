import os
import base64
from dotenv import load_dotenv
from flask import Flask, request, jsonify, Blueprint

import io
import warnings
from PIL import Image, ImageDraw, ImageFont, ImageColor
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
import openai
import cv2
import json

load_dotenv()


os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'
os.environ['STABILITY_KEY'] = os.getenv('STABLE_DIFFUSION_API')
os.environ['OPENAI_API'] = os.getenv('OPEN_AI_API')

stability_api = client.StabilityInference(
    key=os.environ['STABILITY_KEY'],
    verbose=True,
    engine="stable-diffusion-xl-beta-v2-2-2",
)


app = Flask(__name__)
getGlimpse = Blueprint('getGlimpse', __name__)




# ==== Helper Functions ====


# Converting user text to conversation using gpt 
# @getGlimpse.route('/getinstruction', methods=['POST'])
def fetch_instructions_and_captions(prompt):

    openai.api_key = os.environ['OPENAI_API']
    # article = request.get_json()['userInput']

    # Call the OpenAI API to generate a response
    # prompt = "You are a comic book author specialized in digital art,You have to make give instructions to panel for drawing, the content is :"+article+"\n\nPlease write detailed drawing instructions and a one-sentence short caption for the upto 4 panels of a new silent comic book page.Give your response as a JSON array like this: Array<{ panel: number; instructions: string; caption: string}>.Be brief in your 4 instructions and captions, don't add your own comments. Be straight to the point, and never reply things like \"Sure, I can..\" etc."

    response = openai.ChatCompletion.create(

        model="gpt-3.5-turbo",

        messages=[{

            "role": "system",

            "content": "You are a fun yet knowledgable assistant."

        }, {

            "role": "user",

            "content": prompt

        }],

        temperature=0.6,

        max_tokens=1000)

    # Process the response to extract speech and person information

    res = json.loads(response.choices[0].message.content)

    # Return the generated speech and person information
    return res


# Generate map in the format of {0: "speech", 1: "speech", ...} and {0: "person", 1: "person", ...}

def generate_map_from_text(text):

    d = {}

    who_spoke = {}

    dialogue = []

    speak = []

    l = text.split("\n")

    for word in l:

        i = 0

        if 'Scene' not in word and 'Act' not in word:

            if ':' in word:

                dialogue.append((word.split(':')[1]))

                speak.append((word.split(':')[0]))

        for i in range(len(dialogue)):

            d[i] = dialogue[i]

            who_spoke[i] = speak[i]

    return (d, who_spoke)


# Create an image from the generated speech and person information using the Stable Diffusion API

def stable_diff(prompt,number):
    
        answer = stability_api.generate(
    
            prompt=prompt,
    
            seed=992446758,
    
            steps=int(30),
    
            cfg_scale=int(8),
    
            width=512,
    
            height=512,
    
            samples=1,
    
            sampler=generation.SAMPLER_K_DPMPP_2M
    
        )
    
        # Check if the folder exists, create it if necessary
    
        folder_path = "./images"
    
        # Save the generated image to the folder
    
        a = 0
    
        # print(answer)
    
        for resp in answer:
    
            for artifact in resp.artifacts:

                try :
    
                    if artifact.finish_reason == generation.FILTER:
        
                        return 'Something went wrong'
        
                    if artifact.type == generation.ARTIFACT_IMAGE:
        
                        image_path = os.path.join(folder_path, f"{number}.png")
        
        
                        # server\images\0.png
        
                        img_binary = io.BytesIO(artifact.binary)
        
                        img = Image.open(img_binary)
        
                        img.save(image_path)
        
                        return image_path
                except :
                    return 'Something went wrong'

# def stable_diff(prompt cfg, step):

#     answer = stability_api.generate(
#         prompt=prompt,

#         seed=992446758,

#         steps=int(10),

#         cfg_scale=int(8),

#         width=512,

#         height=512,

#         samples=1,

#         sampler=generation.SAMPLER_K_DPMPP_2M

#     )

#     # Check if the folder exists, create it if necessary

#     folder_path = "./images"
#     # Save the generated image to the folder
#     a=0

#     print(answer)

#     for resp in answer:

#         for artifact in resp.artifacts:

#             if artifact.finish_reason == generation.FILTER:

#                 warnings.warn(

#                     "Your request activated the API's safety filters and could not be processed."

#                     "Please modify the prompt and try again.")

#             if artifact.type == generation.ARTIFACT_IMAGE:

#                 image_path = os.path.join(folder_path, f"{a}.png")
#                 a+=1
#                 # comicify_server\images\0.png

#                 img_binary = io.BytesIO(artifact.binary)

#                 img = Image.open(img_binary)

#                 img.save(image_path)

#                 return image_path




# Add text to the generated image using OpenCV and PIL

def add_text_to_image(image_path, caption, file_number):

    # input should be an image and corresponding text needs to be added after padding

    #text= text

    # can probably ask for colour of padding, colour of font for each.

    image = Image.open(image_path)

    right_pad = 0

    left_pad = 0

    top_pad = 50

    bottom_pad = 0

    width, height = image.size

    new_width = width + right_pad + left_pad

    new_height = height + top_pad + bottom_pad

    result = Image.new(image.mode, (new_width, new_height), (255, 255, 255))
    result.paste(image, (left_pad, top_pad))

    font_type = ImageFont.truetype("./font/animeace2_reg.ttf", 12)

    draw = ImageDraw.Draw(result)

    draw.text((10, 0), caption, fill='black', font=font_type)

    result.save(f"./images/{file_number}.png")

    border_img = cv2.imread(f"./images/{file_number}.png")

    borderoutput = cv2.copyMakeBorder(
        border_img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    cv2.imwrite(f"./images/{file_number}.png", borderoutput)


# ==== Routes ====

@getGlimpse.route('/', methods=['GET'])
def test():
    return 'The server is running!'


@getGlimpse.route('/getGlimpse_course', methods=['POST'])
def generate_comic_from_text():

    course = request.get_json()['userInput']

    prompt = "You are a comic book author specialized in digital art,You have to  give instructions to panel for drawing, the content of the comic is about :"+course+"\n\nPlease write detailed drawing instructions and a one-sentence short caption for upto 5 panels of a new silent comic book page.Give your response as a JSON array like this: Array<{ panel: number; instructions: string; caption: string}>.Be brief in your upto 6 instructions and captions, don't add your own comments. Be straight to the point, and never reply things like \"Sure, I can..\" etc."

    response = fetch_instructions_and_captions(prompt)

    print(response)

    image_dict = {}
    for i in range(len(response)):

        image_path = stable_diff(response[i]['instructions'],i)

        if(image_path == 'Something went wrong'):
            print("invalid prompt : ",response[i]['instructions'] )
            continue

        add_text_to_image(f"./images/{i}.png", response[i]['caption'], i)

        # Read the image file
        with open(f"./images/{i}.png", 'rb') as image_file:  
            image_data = image_file.read()
        
        # Convert the image data to Base64
        encoded_image = base64.b64encode(image_data).decode('utf-8')
        
        # Create a dictionary to hold the image data
        image_dict[ f'image {i}' ] = encoded_image
         
    return jsonify(image_dict)


# server\images\0.png
  
@getGlimpse.route('/getimg', methods=['POST'])
def getimg():

    for i in range(5):
         # Read the image file
        with open(f"./images/"+i+".png", 'rb') as image_file:  
            image_data = image_file.read()
        
        # Convert the image data to Base64
        encoded_image = base64.b64encode(image_data).decode('utf-8')
        
        # Create a dictionary to hold the image data
        image_dict = { 'image': encoded_image }

   
     
    return jsonify(image_dict)

# if __name__ == "__main__":

#     app.run(debug=True,host='0.0.0.0')
