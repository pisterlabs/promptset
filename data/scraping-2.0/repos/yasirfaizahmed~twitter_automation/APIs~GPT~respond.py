import openai as ai
import os

from base.base_step import BaseStep

secret = os.environ['API_KEY']
ai.api_key = secret


# Set the prompt
class generate_gpt3_response(BaseStep):
    """
    Disallowed usage
    We don't allow the use of our models for the following:

    * Illegal activity
    * Child Sexual Abuse Material or any content that exploits or harms children
    * Generation of hateful, harassing, or violent content
    * Generation of malware
    * Activity that has high risk of physical harm
    * Activity that has high risk of economic harm
    * Fraudulent or deceptive activity
    * Adult content, adult industries, and dating apps
    * Political campaigning or lobbying
    * Activity that violates people's privacy
    * Engaging in the unauthorized practice of law, or offering tailored legal advice without a qualified person reviewing the information
    * Offering tailored financial advice without a qualified person reviewing the information
    * Telling someone that they have or do not have a certain health condition, or providing instructions on how to cure or treat a health condition
    * High risk government decision-making

    Query OpenAI GPT-3 for the specific key and get back a response
    :type user_text: str the user's text to query for
    :type print_output: boolean whether or not to print the raw output JSON
    """
    def __init__(self, user_prompt: str,
                 engine='text-davinci-003',
                 temperature=0.5,
                 max_tokens=50,
                 **kwargs):
      super().__init__(**kwargs)
      self.user_prompt = user_prompt
      self.engine = engine
      self.temperature = temperature
      self.max_tokens = max_tokens

    def Do(self) -> None:
      completions = ai.Completion.create(engine=self.engine,            # Determines the quality, speed, and cost.
                                         temperature=self.temperature,  # Level of creativity in the response
                                         prompt=self.user_prompt,       # What the user typed in
                                         max_tokens=self.max_tokens,    # Maximum tokens in the prompt AND response
                                         n=1,                           # The number of completions to generate
                                         stop=None)
      self.response.data = completions
      self.response.ok = True

    def CheckCondition(self):
      return self.response.ok


if __name__ == '__main__':
    prompt = 'tell me about the reality of corrupt indian politics'
    response = generate_gpt3_response(prompt)()
    print(response)

    print(response)
