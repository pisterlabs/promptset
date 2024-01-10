import boto3
import os
import openai
from uuid import uuid4

class PollyTTS:
    def __init__(self):
        self.region_name = "us-east-1"
        self.client:boto3.client = boto3.client("polly", region_name=self.region_name)
        self.voice_id = "Matthew"
        self.language_code = "en-US"
        self.output_format = "mp3"
        self.engine = "neural"

    def synthesize(self, text:str) -> bytes:
        response = self.client.synthesize_speech(Text=text, VoiceId=self.voice_id, LanguageCode=self.language_code, OutputFormat=self.output_format, Engine=self.engine, TextType="text")
        cur_uuid = uuid4()
        with open(f"output/voice_{cur_uuid}.mp3", "wb") as f:
            f.write(response["AudioStream"].read())
            f.close()

        with open(f"output/voice_{cur_uuid}.txt", "w") as f:
            f.write(text)
            f.close()
    
    def synthesize_random_story(self):
        text = ChatGPT().get_random_short_fun_story()
        self.synthesize(text)



class ChatGPT():
    def __init__(self):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        # asistant = {"role": "system", "content": "Make message more exciting."}
        self.msgs = []
        self.model = "gpt-3.5-turbo"

    def get_random_short_fun_story(self):
        user = {"role": "user", "content": "Tell me a very short fun story."}
        self.msgs.append(user)
        completion = openai.ChatCompletion.create(
        model=self.model,
        messages=list(self.msgs)
        )

        return completion.choices[0].message.content
    

if __name__ == "__main__":
    tts = PollyTTS()
    tts.synthesize_random_story()
    