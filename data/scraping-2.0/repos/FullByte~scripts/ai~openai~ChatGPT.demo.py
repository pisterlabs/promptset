import openai
openai.api_key = ""

# list engines
engines = openai.Engine.list()

# print the first engine's id
print(engines.data[0].id)

# create a completion
completion = openai.Completion.create(engine="davinci", prompt="Explain how to unscrew a screw")

# print the completion
print(completion.choices[0].text)

