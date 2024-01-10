from fastapi import FastAPI, HTTPException, UploadFile, File
import base64
import os
import uvicorn
from drug_interactions import *
import json

app = FastAPI()

@app.post("/drug-interactions")
async def drug_interactions(drugs: dict):

    try:
        drugs = drugs["drugs"]
        if not isinstance(drugs, list):
            raise HTTPException(status_code=400, detail="Invalid payload. 'drugs' should be a list.")

        interactions = describe_interactions(drugs)

        if not interactions["pairs"]:

            return {
                "drug_interactions": False,
                "explanation": "No interactions found"
            }

        explanation = explain_interactions(interactions)

        return {
            "drug_interactions": True,
            "explanation": explanation
        }
    except:
        raise HTTPException(status_code=400, detail="Invalid payload. 'drugs' should be a list.")

@app.post("/image-extract")
async def upload_and_process_image(image: UploadFile = File(...)):
    image_bytes = await image.read()
    print("read image")

    # Convert binary image data to base64
    base64_rotated_image = base64.b64encode(image_bytes).decode('utf-8')
    print("converted to base64")

    # You can now use base64_rotated_image for further processing or sending to the OpenAI API
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You will take a medicine box image from the user, and you will extract the text and the colors in hex from the image and you will describe the medicine and how to use it.\nYour response should be a json.\n RESPOND ONLY WITH JSON WITHOUT ANY OTHER TEXT.\n { 'text': 'text here', 'colors': ['colors in hex here #FFFFFF'], 'description': 'description here' }"
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "the image I provided is a picture of a medicine box respond with json with key 'text' text is any text visible inside the image and key colors is any colors in hex visible in the image, and a key description with any description of the medicine and how to use. like this {'text': 'text here', 'colors': ['colors', 'here', '#FFFFFF'], 'description': 'description here'}.\nONLY RESPOND WITH JSON WITHOUT ANY OTHER TEXT."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpg;base64,{base64_rotated_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }
    headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}"
        }

    print("sending request to openai")

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        print("got response from openai")

        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="error")

        response = response.json()

        if "choices" not in response:
            raise HTTPException(status_code=400, detail="error in response")

        response = response["choices"][0]["message"]["content"]

        first_curly = response.find("{")
        last_curly = response.rfind("}")

        response = response[first_curly:last_curly+1]

        print("got response from openai with text")

        json_response = json.loads(response)

        return json_response
    except:
        raise HTTPException(status_code=400, detail="error")


uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))