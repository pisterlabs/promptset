# In this codebase, I have a function that transform the Met Office JSON data to CSV
# Here, I played around to see if an LLM could do this transform directly without needing code.


import json
import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from dotenv import load_dotenv
from langchain.callbacks import get_openai_callback
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate
from log_config import get_logger

load_dotenv(".env")

log = get_logger()

open_ai_api_key = os.getenv("OPENAI_API_KEY")

forecast_template = """
Here is some JSON data.
Each Rep becomes a row.
Each Rep has a '$' field which represents the "minutes from midnight" from the Period date.
You have to calculate the actual date-time using the Period date and "minutes from midnight".
For example, if the Period date is 2023-07-10, and the $ value is 540, this represents 2023-07-10 09:00:00.
---------
{json}
---------
Include the CSV on a single line.
Include the header row with all field names and units.
Include the calculated DateTime for each row.
"""


custom_domain_template = """
Here is some JSON data.
Each Block becomes a row.
Each Block field code can be mapped to a meaningful label in the Dict Map.
For example, field with code 'A' becomes column 'Feels'.
Each Block has a 'tm' field which represents the "minutes from midnight" from the Segment date.
You have to calculate the actual date-time using the Segment date and "minutes from midnight".
For example, if the Segment date is 2023-07-10, and the 'tm' value is 540, this represents 2023-07-10 09:00:00.
Map the 'H' field like follows: 
- 'GO' maps to 'Good'
- 'VG' maps to 'Very Good'
- 'P' maps to 'Poor'
- 'A' maps to 'Average'
- 'F' maps to 'Unknown'
---------
{json}
---------
Include the CSV on a single line.
Include the header row with all field names.
Include the calculated DateTime for each row.
"""

question = """
Convert the data to CSV format. 
"""

test_data_sets = {
    "weather_forecast_full": {
        "file_path": "data/met_office/sample_forecast_data.json",
        "prompt_template": forecast_template,
    },
    "weather_forecast_slim": {
        "file_path": "experiments/sample_forecast_data_slim.json",
        "prompt_template": forecast_template,
    },
    "custom_domain": {
        "file_path": "experiments/sample_data_madeup.json",
        "prompt_template": custom_domain_template,
    },
}

active_data_set = "custom_domain"  # Change me
active_prompt_template = test_data_sets[active_data_set]["prompt_template"]
active_data_file = test_data_sets[active_data_set]["file_path"]


chat_prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(active_prompt_template),
    ],
    input_variables=["json"],
)


# Create the LLM reference
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", temperature=0, openai_api_key=open_ai_api_key
)

# Create the chains
chain = LLMChain(llm=llm, prompt=chat_prompt, output_key="result", verbose=True)


with open(active_data_file) as file:
    file_contents = file.read()
    json_obj = json.loads(file_contents)

# Execute LLM chain
with get_openai_callback() as cb:
    response = chain(
        {
            "json": json_obj,
        },
        return_only_outputs=True,
    )
    log.debug(response)

    print(response["result"])

    log.debug(cb)
