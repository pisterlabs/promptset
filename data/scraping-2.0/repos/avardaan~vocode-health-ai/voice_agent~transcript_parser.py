import json
import os
import openai
from voice_agent.logger import logger

# TODO: IMPROVE ERROR HANDLING
def get_patient_data_from_transcript(call_transcript: str, structured_output: dict):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    structured_output_json = json.dumps(structured_output)
    openai_chatcompletion_args = {
        "model": "gpt-3.5-turbo",
        "temperature": 0,
        "messages": [
            {
                "role": "system",
                "content": f"You will be provided an unstructured call transcript between a bot and a human. \
                    You will also be given an output JSON shape with null values. \
                    Your task is to parse the call transcript and output a JSON in the provided shape with the \
                    appropriate values filled based on the human's responses. Values that are not provided by the \
                    human may remain null.\nOutput JSON shape - {structured_output_json}",
            },
            {
                "role": "user",
                "content": call_transcript,
            },
        ],
    }

    try:
        response = openai.ChatCompletion.create(**openai_chatcompletion_args)
    except Exception as e:
        logger.error(f"Error in openai.ChatCompletion.create - {e}")
        return None

    response_message: str = (
        response.choices[0].message.content if response.choices[0] else ""
    )
    parsed_json = json.loads(response_message)
    return parsed_json
