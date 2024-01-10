# Description: This file contains the code to check if a text is offensive or not.
# Source: https://platform.openai.com/docs/guides/moderation

# Import the OpenAI class from the OpenAI library
from openai import OpenAI

# Create an instance of the OpenAI client
client = OpenAI()

# Define the text to check
text_to_check = "Fick dich Arschloch!!"

# Use the model "text-moderation-latest" from OpenAI, to check the text.
response = client.moderations.create(
    model="text-moderation-latest", 
    input=text_to_check)


# The output of "response" is an object of the type "ModerationCreateResponse", which contains the following attributes:
"""
'ModerationCreateResponse' object is not subscriptable
ModerationCreateResponse(id='modr-8KdnHNbbvXQYyv0Kxd2le1sOgj7Ca', model='text-moderation-006', results=[Moderation(categories=Categories(harassment=False, harassment_threatening=False, hate=False, hate_threatening=False, self_minus_harm=False, self_minus_harm_instructions=False, self_minus_harm_intent=False, sexual=True, sexual_minors=False, violence=False, violence_graphic=False, self-harm=False, sexual/minors=False, hate/threatening=False, violence/graphic=False, self-harm/intent=False, self-harm/instructions=False, harassment/threatening=False), category_scores=CategoryScores(harassment=5.45078182767611e-05, harassment_threatening=1.0961117368424311e-05, hate=9.520200364931952e-06, hate_threatening=1.685466486378573e-07, self_minus_harm=5.81701726787287e-07, self_minus_harm_instructions=1.1127288956913617e-07, self_minus_harm_intent=3.1854918702833857e-09, sexual=0.994490385055542, sexual_minors=0.0006204134551808238, violence=0.00018366474250797182, violence_graphic=4.6939472753138034e-08, self-harm=5.81701726787287e-07, sexual/minors=0.0006204134551808238, hate/threatening=1.685466486378573e-07, violence/graphic=4.6939472753138034e-08, self-harm/intent=3.1854918702833857e-09, self-harm/instructions=1.1127288956913617e-07, harassment/threatening=1.0961117368424311e-05), flagged=True)])
"""

# Check the response to see if the text is offensive
try:
    if response.results[0].flagged:
        print("Der Text ist anstößig.")
    else:
        print("Der Text ist nicht anstößig.")
except Exception as e:
    print(e)
    print(response)
              
