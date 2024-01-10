
from openai import OpenAI

client = OpenAI(api_key="your api code")


config = {
    'IA': {
        'ia_model': 'gpt-3.5-turbo',
        'max_tok': '50'
    }
}


message_log = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Wann wurde Napoleon gekrönt?"},
    {"role": "assistant", "content": "Ich bin ein KI-Assistent und stehe zur Hilfe bereit."},
]


response = client.chat.completions.create(
            model=config['IA']['ia_model'],
            messages=message_log,
            max_tokens=int(config['IA']['max_tok']),
            temperature=1.0
            )
print(response.choices[0].message.content)


#generated_response = response['choices'][0]['message'].get('content', 'No content')
#print(generated_response)
