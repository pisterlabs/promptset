import openai
secret_key = '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'
prompt = 'Give me an outline for a Python course'

# Call openai
openai.api_key = secret_key

# create a complete response based on the prompt
output = openai.Completion.create(
    # what model we use?
    model='text-davinci-003',
    prompt=prompt,
    # The max tokens that openAI can give to me? how many characters?
    # 1 token ~= 4 chars, 100 tokens ~= 75 words
    max_tokens=100,
    temperature=0
)

output_text = output['choices'][0]['text']

print(output)
print(output_text)
