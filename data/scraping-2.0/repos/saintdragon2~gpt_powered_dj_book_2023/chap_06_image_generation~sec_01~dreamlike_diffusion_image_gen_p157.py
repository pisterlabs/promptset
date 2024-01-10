import openai

# openai.api_key = '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'
from api_keys import openai_api_key # API key가 github에 올라가면 폐기되기 때문에 따로 import 했습니다.
openai.api_key=openai_api_key  # API key가 github에 올라가면 폐기되기 때문에 따로 import 했습니다.

def text_to_image_prompt_generator(song_title, artist): 
    response=openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        top_p=0.1,
        temperature=0.2,
        messages=[
            {"role": "system", "content":"You are an AI assistant designed to generate prompts for text-to-image models. When a user provides a song title and artist, you should summarize the song's lyrics in a single English sentence, and indicate its genre and mood."}, 
            {"role":"user", "content": f'{song_title} - {artist}'}
        ] 
    ) 
 
    return response.choices[0].message.content


if __name__ == '__main__':
    r=text_to_image_prompt_generator('24K Magic', 'Bruno Mars')
    print(r)