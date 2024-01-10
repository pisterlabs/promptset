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

prompt = "\"\"\"\nCreate an array of weather temperatures for Paris \n\"\"\""

response = openai.Completion.create(
        model = 'text-davinci-002',
        prompt = prompt,
        max_tokens = 256,
        temperature = 0,
        top_p = 1,
        frequency_penalty = 0,
        presence_penalty = 0
    )

print(response)
