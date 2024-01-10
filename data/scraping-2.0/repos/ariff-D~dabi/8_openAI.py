import openai
openai.api_key = "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"
promt = "what is full form of CPU"
res = openai.Completion.create(engine="davinci", prompt = promt, max_token = 50)
genTxt = res.choices[0].text.strip()
print('Result:\n',genTxt)

"""OUTPUT:-
   Result:
   ?
   A full form of CPU is Central Processing Unit which is also commanly
"""