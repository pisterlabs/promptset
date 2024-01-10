import openai
openai.api_key='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'
query="Who is the prime minister of India"
completions=openai.Completion.create(engine='text-davinci-002',prompt=query,max_tokens=250)
message=completions.choices[0].text
message2=message.splitlines()
answer=message2[-1]
print("printing message")
print(message)
print(message2)
print(answer)