# SOURCES
# Fastapi app: https://gist.github.com/SidJain1412/205f3e9ab54c24144599ec625591bd5b
# Modal deployment: https://github.com/modal-labs/modal-examples/blob/main/07_web_endpoints/fastapi_app.py


# Import the required libraries
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import openai
import os
import sys
from pydantic import BaseModel
from modal import Image, Stub, asgi_app, Secret

# Set up the Modal deployment
stub = Stub("example-fastapi-app")

# Define the image for deployment, installing necessary libraries
image = (
    Image.debian_slim()
    .pip_install(
        "fastapi",
        "openai",
        "pydantic"
    )
)

# Create a FastAPI application
app = FastAPI(
    title="Simple ChatGPT API",
)

# Add CORS middleware to the application to handle Cross Origin Resource Sharing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define a Pydantic model for the data the API will receive
class Item(BaseModel):
    prompt: str

# Define a function to get a response from OpenAI based on a prompt
def get_response_openai(prompt):
    try:
        prompt = prompt
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            n=1,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            messages=[
                {"role": "system", "content": "You are an expert creative marketer. Create a campaign for the brand the user enters. Respond in markdwon format."},
                {"role": "user", "content": prompt},
            ],
        )
    except Exception as e:
        print("Error in creating campaigns from openAI:", str(e))
        return 503
    return response["choices"][0]["message"]["content"]

# Define a route that will respond to a GET request with a simple message
@app.get("/ping")
async def ping():
    return "pong"

# Define a route that will respond to a POST request with a marketing campaign based on the received data
@app.post("/campaign")
async def handle_campaign(item: Item):
    return {"campaign_text" : get_response_openai(item.prompt)}

# Set up the deployment details for the Modal platform
@stub.function(
    image=image,
    secret=Secret.from_name("openai"),
    keep_warm=1
)

# Define the ASGI application for deployment
@asgi_app()
def fastapi_app():
    openai.api_key = os.environ["OPENAI_API_KEY"]
    return app

# Deploy the application if this script is run directly
if __name__ == "__main__":
    stub.deploy("app")