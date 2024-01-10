import os
import json

from openai import OpenAI
from openai.types.chat.completion_create_params import ResponseFormat

from csec_data_analytics_app.models import Vulnerability, VulnerabilityImpact


class ChatCompletionManager:
    def __init__(self):
        self.client = OpenAI()
        self.client.api_key = os.environ.get('OPENAI_API_KEY')
        self.model = "gpt-4-1106-preview"
        self.response_format = ResponseFormat(type="json_object")
        self.FEATURE_KEY = "impact_type"

    def extract_vulnerability_features(self):
        # Provides an arbitrary sampling of 5 documents. Use limit instead of aggregate for consistent input.
        vulnerabilities = Vulnerability.objects.limit(10)
        for vulnerability in vulnerabilities:
            response = self._get_chat_completion(vulnerability.description)
            response_dict = json.loads(response)
            print(f"Description: {vulnerability.description}\n\t"
                  f"Extracted feature {self.FEATURE_KEY} is {response_dict[self.FEATURE_KEY]}")
            vulnerability_impact = VulnerabilityImpact(
                impacts=response_dict[self.FEATURE_KEY],
                validated=False
            )
            vulnerability.vulnerability_impact = vulnerability_impact
            vulnerability.save()

    def _get_chat_completion(self, vulnerability_description):
        impact_options = ['remote_code_execution', 'data_disclosure', 'denial_of_service', 'privilege_escalation',
                          'unknown']
        prompt = f"Using the key {self.FEATURE_KEY}, what is the potential impact of the vulnerability described in " \
                 f"the vulnerability description.\n"
        prompt += f"Use any of the following to categorize the impact options: {impact_options}\n"
        prompt += f"Output the impact options as a list.\n"
        prompt += f"If there is not reasonable confidence in the impact, then categorize the impact as 'unknown'\n"
        prompt += f"If the vulnerability allows read access to data, then include 'data_disclosure' in the impact " \
                  f"options\n"
        prompt += f"Vulnerability description: {vulnerability_description}"

        messages = [
            {"role": "system", "content": "You are a JSON encoder for the NVD"},
            {"role": "user", "content": prompt}
        ]
        response = self.client.chat.completions.create(model=self.model, response_format=self.response_format,
                                                       messages=messages)
        response_dict = response.choices[0].message.content
        return response_dict
