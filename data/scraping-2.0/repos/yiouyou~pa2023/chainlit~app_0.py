from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
import pandas as pd
import chainlit as cl
import io

from dotenv import load_dotenv
load_dotenv()


# Create an OpenAI object.
llm = OpenAI(temperature=0)


def create_agent(data: str, llm):

    # Create a Pandas DataFrame agent.
    return create_pandas_dataframe_agent(llm, data, verbose=False)


@cl.on_chat_start
async def on_chat_start():

    # Sending an image with the local file path
    elements = [
    cl.Image(name="image1", display="inline", path="./good_day.jpg")
    ]
    await cl.Message(content="Hello there, Welcome to AskAnyQuery related to Data!", elements=elements).send()

    files = None

    # Waits for user to upload csv data
    while files == None:
        files = await cl.AskFileMessage(
            content="Please upload a csv file to begin!", accept=["text/csv"], max_size_mb= 100
        ).send()

    # load the csv data and store in user_session
    file = files[0]
    csv_file = io.BytesIO(file.content)
    df = pd.read_csv(csv_file, encoding="utf-8")

    # creating user session to store data
    cl.user_session.set('data', df)

    # Send response back to user
    await cl.Message(
        content=f"`{file.name}` uploaded! Now you ask me anything related to your data"
    ).send()


@cl.on_message
async def main(message: str):

    # Get data
    df = cl.user_session.get('data')

    # Agent creation
    agent = create_agent(df, llm)

    # Run model 
    response = agent.run(message)

    # Send a response back to the user
    await cl.Message(
        content=response,
    ).send()



##### Custom API endpoints
from chainlit.server import app
from fastapi import Request
from fastapi.responses import (
    HTMLResponse,
)

@app.get("/hello")
def hello(request: Request):
    print(request.headers)
    return HTMLResponse("Hello World")

