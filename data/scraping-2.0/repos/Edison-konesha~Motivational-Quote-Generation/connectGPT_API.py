import openai
import configparser

# Read the API key from the config.ini file
config = configparser.ConfigParser()
config.read("config.ini")
api_key = config.get("openai", "API_KEY")
# print("API Key:", api_key)

openai.api_key = api_key  # private api key accessed by openAI


# emotions and related prompts to send to the GPT api so that it can generate a quote
def generate_prompt(emotion):
    if emotion == 'Happy':
        return 'Generate a motivational quote for someone who is happy.'
    elif emotion == 'Sad':
        return 'Generate a motivational quote for someone who is feeling sad.'
    elif emotion == 'Angry':
        return 'Generate a motivational quote for someone who is angry.'
    elif emotion == 'Scared':
        return 'Generate a motivational quote for someone who is scared.'
    elif emotion == 'Anxiety':
        return 'Generate a motivational quote for someone who is feeling anxious.'
    elif emotion == 'Guilt':
        return 'Generate a motivational quote for someone who is feeling guilty.'
    elif emotion == 'Loneliness':
        return 'Generate a motivational quote for someone who is feeling lonely.'
    elif emotion == 'Panicked':
        return 'Generate a motivational quote for someone who is feeling panicked.'
    elif emotion == 'Unknown':
        return 'Generate a motivational quote for someone who is feeling neutral.'
    else:
        return 'Generate a motivational quote.'


# get and return the response from API and quote generation
def get_gpt_response(prompt, model='text-davinci-002', max_tokens=50):
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=max_tokens,
        n=1,
        stop=None,
        temperature=0.7
    )

    return response.choices[0].text.strip()
