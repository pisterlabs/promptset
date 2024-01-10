import json
import openai
from dotenv import dotenv_values


class GetCountryInfoFromAzureOpenAI:
    config_details = dotenv_values(".env")

    def __init__(self):
        openai.api_type = "azure"
        openai.api_base = self.config_details['OPENAI_API_BASE']
        openai.api_version = self.config_details['OPENAI_API_VERSION']
        openai.api_key = self.config_details["OPENAI_API_KEY"]

    def get_country_info(self, country_name):

        input_prompt = f"Please give me the country_name, capital_state, national_bird, country_population for {country_name} in flat JSON object. country_population should be in BIGINT without separators"

        print("Input prompt:", input_prompt)

        response = openai.Completion.create(
            engine=self.config_details['COMPLETIONS_MODEL'],
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
        print("Country info JSON:", country_info_json)

        # Convert the JSON string to a dictionary
        country_info = json.loads(country_info_json)

        return country_info
