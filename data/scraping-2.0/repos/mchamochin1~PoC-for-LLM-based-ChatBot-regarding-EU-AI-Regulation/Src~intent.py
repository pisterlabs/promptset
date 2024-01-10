# Intent service
#
# When the user submits a question to your application, the intent service’s role is
# to detect the intent of the question. Is the question relevant to your data? Perhaps
# you have multiple data sources: the intent service should detect which is the
# correct one to use. This service could also detect whether the question from the
# user does not respect OpenAI’s policy, or perhaps contains sensitive information.
# This intent service will be based on an OpenAI model in this example.

import openai

INTENT_MODEL_NAME="gpt-3.5-turbo"

class IntentService():
    def __init__(self):
        self.intent_model = INTENT_MODEL_NAME
     
    def get_intent(self, query: str = 'Hello'):
        # call the openai ChatCompletion endpoint
        response = openai.chat.completions.create(model=self.intent_model, messages=[{"role": "user", "content": f'Extract the keywords from the following question: {query}'+'Do not answer anything else, only the keywords.'}])
        return (response.choices[0].message.content)
