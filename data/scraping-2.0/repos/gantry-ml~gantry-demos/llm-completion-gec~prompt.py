import gantry
import os
from dotenv import load_dotenv
from gantry.applications.llm_utils import fill_prompt
import openai
from openai.util import convert_to_dict
load_dotenv()

GANTRY_API_KEY = os.getenv("GANTRY_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

gantry.init(api_key=GANTRY_API_KEY)

my_llm_app = gantry.get_application("my-app-docs")

version = my_llm_app.get_version("prod")
config = version.config
prompt = config['prompt']

def generate(user_input_value):
    values = {
        "user_input": user_input_value
    }
    filled_in_prompt = fill_prompt(prompt, values)
    
    request = {
        "model": "text-davinci-002",
        "prompt": filled_in_prompt,
    }
    results = openai.Completion.create(**request)

    my_llm_app.log_llm_data(
        api_request=request,
        api_response=convert_to_dict(results),
        request_attributes={"prompt_values": values},
        version=version.version_number,
    )

    return results