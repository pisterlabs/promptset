import google.generativeai as palm
import os
from dotenv import load_dotenv
from langchain.llms import VertexAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from kor.extraction import create_extraction_chain, Object, Text
from kor.nodes import Object, Text, Number

load_dotenv()

#def get_response(message: str) -> str:
    #palm.configure(api_key=os.environ['PALM_APIKEY'])
    #response = palm.chat(context="",messages=[message])
    #return response.last

def get_response(message: str) -> str:
    llm = VertexAI()
    schema = Object(
        id="player",
        description=(
            "User is controlling a music player to select songs, pause or start them or play"
            " music by a particular artist."
        ),
        attributes=[
            Text(
                id="song",
                description="User wants to play this song",
                examples=[],
                many=True,
            ),
            Text(
                id="album",
                description="User wants to play this album",
                examples=[],
                many=True,
            ),
            Text(
                id="artist",
                description="Music by the given artist",
                examples=[("Songs by paul simon", "paul simon")],
                many=True,
            ),
            Text(
                id="action",
                description="Action to take one of: `play`, `stop`, `next`, `previous`.",
                examples=[
                    ("Please stop the music", "stop"),
                    ("play something", "play"),
                    ("play a song", "play"),
                    ("next song", "next"),
                ],
            ),
        ],
        many=True,
    )
    chain = create_extraction_chain(llm, schema, encoder_or_encoder_class="json")