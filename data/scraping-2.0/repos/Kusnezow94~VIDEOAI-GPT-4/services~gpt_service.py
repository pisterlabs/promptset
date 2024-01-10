```python
import openai

openai.api_key = 'your-api-key'

def generate_prompt(channel_name, channel_description):
    prompt = f"Generate video ideas for a YouTube channel named {channel_name} which is about {channel_description}."
    return prompt

def generate_ideas(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.5,
        max_tokens=100
    )
    ideas = response.choices[0].text.strip().split('\n')
    return ideas

def generate_script(title):
    prompt = f"Generate a script for a YouTube short video with the title {title}."
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.5,
        max_tokens=500
    )
    script = response.choices[0].text.strip()
    return script

def generate_video_metadata(title):
    prompt = f"Generate a description and hashtags for a YouTube short video with the title {title}."
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        temperature=0.5,
        max_tokens=100
    )
    metadata = response.choices[0].text.strip().split('\n')
    description = metadata[0]
    hashtags = metadata[1:]
    return description, hashtags
```