# %%
import openai
import pandas as pd
from pytube import YouTube
from moviepy.editor import AudioFileClip
import whisper

# %%
class ConferenceSummarizer:
    def __init__(self, api_key, cx, query, num_results, openai_api_key):
        self.api_key = api_key
        self.cx = cx
        self.query = query
        self.num_results = num_results
        self.results = []
        openai.api_key = openai_api_key  # replace with your OpenAI API key
        
    # TODO Get url address => First input
    def download_youtube_audio(self, youtube_video_url):
        self.youtube_video = YouTube(youtube_video_url)
        streams = self.youtube_video.streams.filter(only_audio=True)
        stream = streams.first()
        stream.download(filename=f'{self.youtube_video.title}.mp4')
        return f'{self.youtube_video.title}.mp4'
    
    # TODO Get trim time sequence => Second input
    def trim_audio(self, start_time, end_time):
        # Load the audio file with moviepy
        audio_clip = AudioFileClip(f'{self.youtube_video.title}.mp4')

        # Trim the audio
        trimmed_clip = audio_clip.subclip(start_time, end_time)

        # Write the result to a file
        trimmed_clip.write_audiofile(f'trimmed_{self.youtube_video.title}.mp3')
    
    # TODO Give converted audio => First output
    def transcribe_audio(self):
        model = whisper.load_model('base')
        output = model.transcribe(f'trimmed_{self.youtube_video.title}.mp3')
        return output
    
    # TODO Give transcribed text => Second output
    # TODO Give summary and its pdf => Third output and final
    # TODO Increase token limit?
    # TODO Summary customization? OpenAI models or LangChain models?
    def summarize_transcription(self, transcription):
        # Extract the actual transcription text from the dictionary
        transcription_text = transcription.get('transcription', '')
        
        # Split the transcription into chunks of 3000 tokens each
        tokens = transcription_text.split()
        chunks = [' '.join(tokens[i:i + 3000]) for i in range(0, len(tokens), 3000)]

        summaries = []
        for chunk in chunks:
            # Summarize each chunk using the OpenAI API
            prompt = f"Please provide a bit detailed summary of the following text with keywords max 500 words:\n{chunk}"
            response = openai.Completion.create(
                model="text-davinci-003",
                prompt=prompt,
                max_tokens=1000,
                n=1,
                stop=None,
                temperature=0.5,
            )
            summaries.append(response.choices[0].text.strip())

        # Aggregate the summaries
        aggregated_summary = ' '.join(summaries)

        # Summarize the aggregated summary
        prompt = f"The following text are aggregated summaries of a conference. Aggregate the text and produce an report article like Bloomberg article, max 1000 words with core keywords:\n{aggregated_summary}"
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=2000,
            n=1,
            stop=None,
            temperature=0.5,
        )
        final_summary = response.choices[0].text.strip()

        # Save the final summary to a text file
        with open(f'final_summary_{self.youtube_video.title}.txt', 'w') as f:
            f.write(final_summary)

# %%
# Load the model
summarizer = ConferenceSummarizer(api_key='-', cx='-', query='fomc conference 2023', num_results=3, openai_api_key='-')

# %%
# Step 1: Download the audio from a YouTube video
youtube_video_url = 'https://www.youtube.com/watch?v=ifqyTQ0Ifrw&t=855s'  # replace with your YouTube video URL
summarizer.download_youtube_audio(youtube_video_url)

# %%
# Step 2: Trim the audio
start_time = 1  # replace with the start time in seconds
end_time = 440  # replace with the end time in seconds
summarizer.trim_audio(start_time, end_time)

# %%
# Step 3: Transcribe the audio
transcription = summarizer.transcribe_audio()

# %%
# Step 4: Summarize the transcription
summarizer.summarize_transcription(transcription)

# %%
# def main():
#     pass
    
# if __name__ == '__main__':
#     main()
    
# TODO Powerpoint: expected improvement and framework => LangChain, Customized vector DB on Cloudserver => Possible direct implementation
