import openai
import os
from googletrans import Translator
from dotenv import load_dotenv # Add
load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
model_id = 'gpt-3.5-turbo'

translator = Translator()

def ChatGPT_conversation(conversation):
    response = openai.ChatCompletion.create(
        model=model_id,
        messages=conversation
    )
    
    # api_usage = response['usage']
    # print('Total token consumed: {0}'.format(api_usage['total_tokens']))
    # stop means complete
    # print(response['choices'][0].finish_reason)
    # print(response['choices'][0].index)
    conversation.append({'role': response.choices[0].message.role, 'content': response.choices[0].message.content})
    return conversation

conversation = []
conversation.append({'role': 'system', 'content': 'How may I help you?'})
conversation = ChatGPT_conversation(conversation)
print(translator.translate(text=
    '{0}: {1}\n'.format(
            conversation[-1]['role'].strip(),
            conversation[-1]['content'].strip()
        ),
        dest='zh-tw'
        ).text
    )

while True:
    user_input = input('User:')
    prompt = translator.translate(
        text=user_input,
        dest='en').text
    conversation.append({'role': 'user', 'content': prompt})
    conversation = ChatGPT_conversation(conversation)
    print(translator.translate(text=
    '{0}: {1}\n'.format(
            conversation[-1]['role'].strip(),
            conversation[-1]['content'].strip()
        ),
        dest='zh-tw'
        ).text
    )