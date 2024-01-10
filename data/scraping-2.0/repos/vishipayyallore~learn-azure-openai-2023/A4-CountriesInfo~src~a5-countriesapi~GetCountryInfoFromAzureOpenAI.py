import json
import openai
from env_config import get_config_value


class GetCountryInfoFromAzureOpenAI:
    def __init__(self):
        openai.api_type = "azure"
        openai.api_base = get_config_value('OPENAI_API_BASE')
        openai.api_version = get_config_value('OPENAI_API_VERSION')
        openai.api_key = get_config_value("OPENAI_API_KEY")

    def get_country_info(self, country_name):
        input_prompt = f"Please give me the country_name, capital_state, national_bird, country_population for {country_name} in flat JSON object. country_population should be in BIGINT without separators"

        response = openai.Completion.create(
            engine=get_config_value('COMPLETIONS_MODEL'),
            prompt=input_prompt,
            temperature=1,
            max_tokens=300,
            top_p=0.5,
            frequency_penalty=0,
            presence_penalty=0,
            best_of=1,
            stop=None)

        # Assuming the response.choices[0].text is a JSON string
        country_info_json = response.choices[0].text

        # Convert the JSON string to a dictionary
        country_info = json.loads(country_info_json)

        return country_info
