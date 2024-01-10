import os
from pytube import YouTube
from google.cloud import speech
import openai
import re

openai.api_key = "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"

class AudioTranscriber:
    def __init__(self,url,audio_path):
        self.url = url
        self.audio_path = audio_path
        # print(self.url,self.audio_path)

    def download_audio(self):
        yt = YouTube(self.url)
        # print(yt)
        video = yt.streams.filter(only_audio=True).first()
        print(video)
        download_path = 'audios/'
        if not os.path.isdir(download_path):
            os.makedirs(download_path)
        out_file = video.download(output_path=download_path)
        base,ext = os.path.splitext(out_file)
        new_file = f"{base}.mp3"
        os.rename(out_file,new_file)
        file_name = os.path.basename(new_file)
        file_path = os.path.basename(download_path,file_name)
        return file_path
    
    # def transcribe_audio(self,file):
    #     client = speech.SpeechClient()
    #     with open(file,"rb") as f:
    #         audio_data = f.read()
    #         transcript = client.transcribe(
    #             audio_data=audio_data,
    #             sample_rate_hertz=44100,
    #             language_code="en-US"
    #         )
    #         return transcript.results[0].alternatives[0].transcript
    
    def transcribe(self):
        # try:
            audio_path = self.download_audio()
            # transcription = self.transcribe_audio(audio_path)
            return "transcription"
        # except Exception as e:
        #     print("Error",e)
        #     return "cannot transcribe"

class Summarizer:
    """
    Summarizer class that uses OpenAI API to generate summaries
    """
    def __init__(self, content):
        self.content = self.replace_non_alphanumeric(content)   
        self.max_tokens = 512
        self.max_len = 7500

    def replace_non_alphanumeric(self, string):
        # Use regex to replace all non-alphanumeric characters with an empty space
        return re.sub(r'[^a-zA-Z0-9]', ' ', string)


    def summarize(self):
        # Split text into chunks that are within the token limit
        chunks = self.split_text(self.content)
        summary = ""

        # Generate summary for each chunk of text
        for chunk in chunks:

            prompt = chunk + "\nTl;dr"

            # Generate summary using OpenAI API.
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=self.max_tokens,
                n=1,
                stop=None,
                temperature=0.7,
            )

            generated_summary = response.choices[0].text.strip()
            summary += generated_summary

        return summary

    def split_text(self, text):
        chunks = []
        for i in range(0, len(text), self.max_len):
            chunks.append(text[i:i+self.max_len])
        return chunks


if __name__ == "__main__":
    url = input("Enter Youtube URL: ")
    transciber = AudioTranscriber(url,"audios/")
    transcibed_text = transciber.transcribe()
    print("\n\n",transcibed_text)
    summary = Summarizer(transcibed_text).summarize()
    print("\n\n", summary)