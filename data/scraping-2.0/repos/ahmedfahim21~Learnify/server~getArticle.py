from flask import Flask , jsonify, Blueprint, request
from flask_cors import CORS, cross_origin
import openai
from dotenv import load_dotenv
import os
import json
from stability_sdk import client
import warnings
from PIL import Image, ImageDraw, ImageFont, ImageColor
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
import io
import random

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
get_Article = Blueprint('get_Article', __name__)


# ==== Helper Functions ====


def stable_diff(paragraph):

    answer = stability_api.generate(
        prompt=f"""

        Create an eye-catching illustration that visually represents the concept of 'Inheritance' in object-oriented programming. 

        """,

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
    a=0

    print(answer)

    for resp in answer:

        for artifact in resp.artifacts:

            if artifact.finish_reason == generation.FILTER:

                warnings.warn(

                    "Your request activated the API's safety filters and could not be processed."

                    "Please modify the prompt and try again.")

            if artifact.type == generation.ARTIFACT_IMAGE:

                image_path = os.path.join(folder_path, f"{a}.png")
                # comicify_server\images\0.png
                a += 1

                img_binary = io.BytesIO(artifact.binary)

                img = Image.open(img_binary)

                img.save(image_path)

                return image_path


def generate_image_prompt(paragraph):
    # paragraph = request.get_json()['userInput']
    openai.api_key = os.environ['OPENAI_API']
    # Generate a prompt for the image
    prompt = "You are a comic book author specialized in digital art,Please write detailed drawing instruction and a one-sentence short caption for a artist of a new silent comic book page for illustrating this" +paragraph+".Give your response as a JSON like this: \`{ instructions: string; caption: string}\.Be brief in your instruction and captions, don't add your own comments. Be straight to the point, and never reply things like \"Sure, I can..\" etc."
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

        max_tokens=150)
    
    res = json.loads(response.choices[0].message.content)

    print(res)
    
    return res['instructions'],res['caption']
    




# @get_Article.route('/get_Article_images',methods=['POST'])
# def embed_article_with_image():
#     article = request.get_json()['userInput']
#     # Divide the article into paragraphs
#     paragraphs = article.split('\n')
    
#     # iterate through each paragraph and add an image to it
#     for i in range(len(paragraphs)):
#         if(random.randint(0,1)==0):
#             continue
#         imageprompt,caption = generate_image_prompt(paragraphs[i])
#         image = stable_diff(imageprompt)
#         paragraphs[i] = paragraphs[i] + '\n' + image + '\n' + caption
              
       
        
#     # Return the article with images
#     return paragraphs
   



def talkToGPT(prompt):

    openai.api_key = os.environ['OPENAI_API']

    # Call the OpenAI API to generate a response

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
    
    res = response.choices[0].message.content
    

    # Return the generated modules
    return res






# CORS(app)
@get_Article.route('/get_Article_topics',methods=['POST'])
def get_article_topics():
    module = request.get_json()['userInput']
    prompt = "What are the topics for " + module + "?.Give your response as a JSON like this: \`{ topics: Array<string> \`}" + "Be straight to the point, and never reply things like \"Sure, I can..\" etc."
    openai.api_key = os.environ['OPENAI_API']

    # Call the OpenAI API to generate a response

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
    
    articles = json.loads(response.choices[0].message.content)


    # Return the generated modules
    return articles['topics']

@get_Article.route('/get_ShortNote',methods=['POST'])
def get_short_note():
    module = request.get_json()['userInput']
    prompt = "Give me a short note within 25 words on " + module + "Be straight to the point, and never reply things like \"Sure, I can..\" etc."
    # Your route logic here
    return talkToGPT(prompt)



@get_Article.route('/get_Article',methods=['POST'])
def get_data():

    article_name = request.get_json()['userInput']
    article_name_prompt = "Give me a detailed article on " + article_name + "Be straight to the point, and never reply things like \"Sure, I can..\" etc."
    # Your route logic here
    return talkToGPT(article_name_prompt)

@get_Article.route('/get_novel_article',methods=['POST'])
def get_novel_article():

    articleContent = request.get_json()['userInput']
    prompt = "Convert this simple article " + articleContent + " \n\n into a interesting story. You can use any engaging story telling format so that the attentiion of the user is high throughout and try to add a suspense at the end so that reader is willing to read the next article. never reply things like \"Sure, I can..\" etc. "

    return talkToGPT(prompt)
