from fastapi import APIRouter
import openai
from json import loads
from os import getenv
from textwrap import dedent
from fastapi import HTTPException
from dotenv import load_dotenv
load_dotenv()

router = APIRouter()

# Load OpenAI API key and model engine from environment variables
openai.api_key = getenv("OPENAI_API_KEY")
model_engine = getenv("OPENAI_MODEL_ENGINE")

ABSA_PROMPT = dedent(
        f"""
        fetch out aspect, descriptor and polarity of each aspect from the following sentence. The polarity should be in the range of 1 to 10. 
        Output format in JSON
        Example json format: 
        
        [{{"aspect": "food", "descriptor": "delicious", "polarity": 10}}, 
        {{"aspect": "toilets", "descriptor": "not clean", "polarity": 1}}]
        """
        )

def gpt_absa_controller(review: str):
    '''
    Generates an aspect-based sentiment analysis (ABSA) response using OpenAI's GPT-3 language model.
    '''
    print(review)
    try:
        completion = openai.Completion.create(
            engine=model_engine,
            prompt= f"{ABSA_PROMPT} \n '{review}'",
            max_tokens=1024,
            n=1,
            stop=None,
            temperature=0.5,
        )

        response = completion.choices[0].text
        raw_json = response.strip()
        json_data = loads(raw_json)
        return json_data
    
    except Exception as e:
        error_msg = f"An error occurred while generating the ABSA response: {str(e)}"
        raise HTTPException(status_code=500, detail=error_msg)
