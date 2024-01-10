import os
import openai
import datetime

from source_material import content

MODEL = 'gpt-4'
# MODEL = 'gpt-3.5-turbo-16k'
TEMPERATURE = 0.2

openai.api_key = os.getenv("OPENAI_API_KEY")

podcast_script = ""
try:
    completion = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a Major League Baseball journalist with a speciality in Sabermetrics"},
            {"role": "user", "content": content.format(datetime.datetime.now())}
        ]
    )
    podcast_script = completion.get("choices")[0].get("message").get("content")
except:
    raise "openai returned garbage"
print("break")
