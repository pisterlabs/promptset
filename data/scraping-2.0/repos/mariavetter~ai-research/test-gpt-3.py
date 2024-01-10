Api_key = "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"


import openai
import time

openai.api_key = Api_key

amount = 267

prompt = "Write only the text of a generic genuine with no placeholders mail."\
    "Response:"


for i in range(amount):
    response = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=70)
    text = response["choices"][0]["text"]
    text = text.replace("\n", "").replace(r"\r\n", "")
    text = text + "\n"
    out = text.replace('\n', '')
    print(f"{i}: {out}")
    with open("normal_gen.txt", "a+") as f:
        f.writelines(text)
    time.sleep(1.3)