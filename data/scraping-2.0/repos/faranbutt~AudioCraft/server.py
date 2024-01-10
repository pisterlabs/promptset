import replicate
from flask import Flask, request, jsonify
from flask_cors import CORS
import argparse
import os
cwd = os.getcwd()
import numpy as np
np.bool = bool
from flask import Flask
from flask_cors import CORS
from flask_restful import Resource
from flask_restful import Api
from flask import jsonify, make_response, send_file
import requests
from bs4 import BeautifulSoup, Comment
import base64
import wave
import array
from io import BytesIO
import openai
openai.api_key = os.environ.get("OPENAI_API_KEY")
print("key:", openai.api_key)

PAT = os.environ.get("PAT")
print("PAT:", PAT)




print("Starting server...")
parser = argparse.ArgumentParser()

# API flag
parser.add_argument(
    "--host",
    default="0.0.0.0",
    help="The host to run the server",
)
parser.add_argument(
    "--port",
    default=os.environ.get("PORT"),
    help="The port to run the server",
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="Run Flask in debug mode",
)

args = parser.parse_args()

app = Flask(__name__)  # static_url_path, static_folder, template_folder...
CORS(app, resources={r"/*": {"origins": "*", "allow_headers": "*"}})











def generate_image_description(image_bytes_64):
    USER_ID = 'salesforce'
    APP_ID = 'blip'

    MODEL_ID = 'general-english-image-caption-blip'
    MODEL_VERSION_ID = 'cdb690f13e62470ea6723642044f95e4'


    from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
    from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
    from clarifai_grpc.grpc.api.status import status_code_pb2

    channel = ClarifaiChannel.get_grpc_channel()
    stub = service_pb2_grpc.V2Stub(channel)

    metadata = (('authorization', 'Key ' + PAT),)

    userDataObject = resources_pb2.UserAppIDSet(user_id=USER_ID, app_id=APP_ID)

    post_model_outputs_response = stub.PostModelOutputs(
        service_pb2.PostModelOutputsRequest(
            user_app_id=userDataObject,  # The userDataObject is created in the overview and is required when using a PAT
            model_id=MODEL_ID,
            version_id=MODEL_VERSION_ID,  # This is optional. Defaults to the latest model version
            inputs=[
                resources_pb2.Input(
                    data=resources_pb2.Data(
                        image=resources_pb2.Image(
                            base64=base64.b64decode(image_bytes_64)
                        )
                    )
                )
            ]
        ),
        metadata=metadata
    )
    if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
        print(post_model_outputs_response.status)
        raise Exception("Post model outputs failed, status: " + post_model_outputs_response.status.description)

    # Since we have one input, one output will exist here
    text = post_model_outputs_response.outputs[0].data.text.raw

    return text



def generate_music_prompt(image_description):
    prompt = f"""There is a new AI called MusicGen which can generate a song given a prompt. Here are some example prompts:
80s pop track with bassy drums and synth
90s rock song with loud guitars and heavy drums
Pop dance track with catchy melodies, tropical percussion, and upbeat rhythms, perfect for the beach 				
A grand orchestral arrangement with thunderous percussion, epic brass fanfares, and soaring strings, creating a cinematic atmosphere fit for a heroic battle. 				
classic reggae track with an electronic guitar solo 				
earthy tones, environmentally conscious, ukulele-infused, harmonic, breezy, easygoing, organic instrumentation, gentle grooves 				
lofi slow bpm electro chill with organic samples 				
drum and bass beat with intense percussions

I have created an app to convert images into songs.
Here are some examples of how an image description can be turned into a prompt:

image description: 5 big rubber ducks are lined up on a street, a man wearing a suit is looking at them
prompt: Quirky and whimsical urban soundtrack featuring playful bass, light electronic beats, and saxophone, capturing the curious essence of a man in a suit observing large rubber ducks on a city street.

image description: A cat wearing a space suit
prompt: Cosmic Electronic Pop with quirky synthesizers and adventurous beats, capturing the whimsy and curiosity of a cat in a space suit.

image description: A colorful Asian landscape with mountains and a waterfall
prompt: Asian-inspired orchestral piece with flowing melodies, featuring traditional instruments like koto, shamisen, and shakuhachi flutes, set against a backdrop of cascading strings and subtle percussion to capture the essence of a colorful landscape with mountains and a waterfall.

The following is a description of the user's image: {image_description}
Create a prompt for MusicGen that represents the image."""

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
            ],
        temperature=0,
        max_tokens=400,
        top_p=0.95,
        stop=[
            "console.log(csv);"
        ]
    )

    text = response.choices[0].message.content
    # text = "90s rock song with loud guitars and heavy drums"
    return text




model = None
processor = None
def generate_music(prompt):
    global model
    global processor
    if model is None:
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
        processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small").cuda()

    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.cuda() for k, v in inputs.items()}
    audio_values = model.generate(**inputs, max_new_tokens=512)
    sampling_rate = model.config.audio_encoder.sampling_rate
    audio_64 = audio_values[0].cpu().numpy()[0].tolist()

    # sampling_rate = 32000
    # audio_64 = ""

    return (sampling_rate, audio_64)













@app.route('/imagedescription', methods=['POST'])
def image_description():
    image_bytes_64 = request.json.get('image')
    description = generate_image_description(image_bytes_64)
    return jsonify({'description': description})

@app.route('/musicgenprompt', methods=['POST'])
def music_prompt():
    image_description = request.json.get('description')
    prompt = generate_music_prompt(image_description)
    return jsonify({'prompt': prompt})

@app.route('/generatemusic', methods=['POST'])
def music_generation():
    # prompt = request.json.get('prompt')
    # sampling_rate, audio_samples = generate_music(prompt)
    
    # # Convert float32 array to 16-bit PCM
    # audio_samples = [int(min(max(sample * 32767, -32768), 32767)) for sample in audio_samples]

    
    # # Create BytesIO object to capture the audio in-memory
    # audio_io = BytesIO()
    
    # # Create WAV file
    # with wave.open(audio_io, 'wb') as wf:
    #     wf.setnchannels(1)
    #     wf.setsampwidth(2)  # 2 bytes for 16-bit PCM
    #     wf.setframerate(sampling_rate)
    #     wf.writeframes(array.array('h', audio_samples).tobytes())
    
    # audio_base64 = base64.b64encode(audio_io.getvalue()).decode('utf-8')
    
    # return jsonify({'sampling_rate': sampling_rate, 'audio': audio_base64})


    prompt = request.json.get('prompt')
    print("sending prompt to replicate:", prompt)
    output = replicate.run(
        "facebookresearch/musicgen:7a76a8258b23fae65c5a22debb8841d1d7e816b75c2f24218cd2bd8573787906",
        input={"model_version": "melody", "prompt": prompt, "duration": 5},
    )
    print("output:", output)
    return jsonify({'audio_url': output})




if __name__ == "__main__":
    app.run(debug=args.debug, host=args.host, port=args.port)