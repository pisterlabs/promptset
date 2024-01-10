print('Hello, world!')
print('Ready to play with an AI?')
print('Oh, we''re gonna have some fun.')
print('Be back in a bit babe. Don''t go anywhere and don''t start without me ;p')
print('OK, let''s have some fun... keep this key a secret for me, K?')
SecretKey = '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'

print('Now let''s just import the positronic network...')
import openai
openai.api_key = SecretKey

MyPrompt = 'Tell me a slogan for a home security company.'
print (MyPrompt)
response = openai.ChatCompletion.create(
    model = "gpt-3.5-turbo"
    #, prompt = MyPrompt #Yeah, I'm kinky with the commas like that baby; you're gonna have some fun.
    #, max_tokens = 200 #~4 tokens per word
    #, temperature = 0 #lower temperature = less random. Scale 0~1 ish
    , messages = [
        {"role": "system", "content": "You are a helpful assistant."}
        , {"role" : "user", "content" : MyPrompt}
    ]
)
output = response['choices'][0]['message']['content']
print(output)