import openai
import time

start_time = time.time()

openai.api_key = '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'

prompt = "This is a test"

system_message = open('system_message_short.txt', 'r').read()

user_message = '''
progression = [C, Dm, Em, F, G, Am, Bdim]
style = [jazz, funk, rnb]
include passing chords = true
number = 3'''

response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
        ]
    )

print(response)
print('Completed in: ', time.time() - start_time)

