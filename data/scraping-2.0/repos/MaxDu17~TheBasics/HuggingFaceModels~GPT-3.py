import openai

with open("apikey.secret") as f:
    openai.api_key = f.readline().strip()

text = "The water is warm, he said to her. Do you want to "

do_top_p = False

kwargs = {"engine": "davinci", "max_tokens": 200}
if do_top_p:
    kwargs['top_p'] = 0.96

r = openai.Completion.create(prompt=f"{text}", **kwargs)
print(text + r['choices'][0].text)

