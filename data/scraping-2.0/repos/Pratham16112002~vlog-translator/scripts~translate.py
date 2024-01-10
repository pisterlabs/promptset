import os
import sys
import openai
import pysrt

openai.api_key = os.getenv("OPENAI_API_KEY")

input_data = sys.stdin.read()
subs = pysrt.from_string(input_data)
for index, subtitle in enumerate(subs):
    print(index, subtitle.text)
prompt_base = (
    "You are going to be a good translator. "
    "Here is a part of the transcript of any video"
    "Translate the following text precisely into Hindi "
    "with the polite and formal style. "
    "Translate from [START] to [END]:\n[START]\n"
)


def translate_text(text):
    prompt = prompt_base
    prompt += text + "[END]\n"
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=999,
        temperature=0,
    )
    return response.choices[0].text.strip()


for index, subtitle in enumerate(subs):
    subtitle.text = translate_text(subtitle.text)
    print(index, subtitle.text)
