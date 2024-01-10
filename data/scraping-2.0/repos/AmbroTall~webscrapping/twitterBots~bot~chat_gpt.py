import openai

openai.api_key = "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"
content = "Hello assistant, I want you to help me monetize my tweeter account by basically generating impressions and attracting more followers to my account."

response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role":"user", "content": content}])

print(response)