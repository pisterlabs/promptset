################################################################################
#                                                                              #
#   Full video tutorial at: https://youtu.be/yNSo6cWpDSM                       #
#                                                                              #
#               Data Science Garage at Youtube.com                             #
#                 (vytautas.bielinskas@gmail.com)                              #
#                                                                              #
################################################################################

import openai

openai.api_key = '<YOUR_OPENAI_API_KEY'  # generate it at: https://beta.openai.com/account/api-keys

prompt = """
Decide whether a Tweet's sentiment is positive, neutral, or negative.

Tweet: I did not like the new Batman movie!
Sentiment:
"""

response = openai.Completion.create(
        model = 'text-davinci-003',
        prompt = prompt,
        max_tokens = 100,
        temperature = 0
    )

print(response)
