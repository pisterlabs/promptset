from flask import Flask, request, send_file, jsonify
# import openai
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
import os
#for image transfer
import io
import base64


model_id = "stabilityai/stable-diffusion-2-1"

app = Flask(__name__)
data = ""

#openai api
# openai.api_key = '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'
# messages = [ {
#     "role": "system",
#     "content": "You are a good assistant but not a human kind."
#     } ]


@app.route("/user_intput", methods=['POST'])
def user_input():
# diffuser
    data =  request.json

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")

    prompt = data #set prompt to user input # add gpt here

    image = pipe(prompt).images[0]
    # image to bytes64 encoding and save to result_collection
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes = image_bytes.getvalue()
    image_base64 = base64.b64encode(image_bytes).decode('utf-8')
    # return {"value":"SERVER get input: "+ data}
    return jsonify({'prompt': prompt, 'image': image_base64})

if __name__ == "__main__":
    app.run(debug=True )

#openai api
    # chat = openai.ChatCompletion.create(
    #         model="gpt-3.5-turbo", messages=messages
    #     )
    # reply = chat.choices[0].message.content
    # print(f"ChatGPT: {reply}")
