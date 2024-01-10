import openai
import yaml

with open('openAI_KEY.yaml', 'r') as file:
    # Load the YAML data
    creds = yaml.safe_load(file)


openai.api_key = creds["OPENAI_API_KEY"]
openai.organization = creds["OPENAI_ORG_ID"]


audio_file= open("./audio.wav", "rb")
transcript = openai.Audio.transcribe(audio_file, initial_prompt='以下是普通話的句子。')
text = transcript.to_dict()['text']
print(text)
