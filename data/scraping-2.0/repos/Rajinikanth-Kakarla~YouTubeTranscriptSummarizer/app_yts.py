# Imports
import json
import streamlit as st
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi
import re
import openai
from streamlit_lottie import st_lottie
from transformers import MarianMTModel, MarianTokenizer

# Set page configuration to a wide layout
st.set_page_config(layout="wide")

# Set the OpenAI API key
openai.api_key = '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'

# Define a function to load a Lottie animation file
class LottieLoader:
    @staticmethod
    def load_lottiefile(filepath: str):
        with open(filepath, 'r') as f:
            return json.load(f)

# Load Lottie animation files
summary = LottieLoader.load_lottiefile('Lottie/summary.json')
trans = LottieLoader.load_lottiefile('Lottie/translate.json')
note = LottieLoader.load_lottiefile('Lottie/note.json')

# Define a class to extract the video ID from a YouTube URL
class VideoIDExtractor:
    @staticmethod
    @st.cache_data
    def get_video_id(url):
        video_id = None
        try:
            video_id = url.split("v=")[1]
            ampersand_position = video_id.find("&")
            if ampersand_position != -1:
                video_id = video_id[:ampersand_position]
        except:
            pass
        return video_id

# Define a class to summarize text
class TextSummarizer:
    @staticmethod
    @st.cache_data
    def summarize_text(text):
        summarizer = pipeline('summarization')
        max_allowed_length = min(len(text), 500)
        min_allowed_length = min(30, max_allowed_length - 1)
        num_iters = int(len(text) / 1000)
        sum_text = []
        for i in range(0, num_iters + 1):
            start = i * 1000
            end = (i + 1) * 1000
            out = summarizer(text[start:end], max_length=max_allowed_length, min_length=min_allowed_length, do_sample=False)
            out = out[0]
            out = out['summary_text']
            sum_text.append(out)
        cleaned_text = re.sub(r'[^A-Za-z\s]+', '', str(sum_text))
        return cleaned_text

# English to French translation
model_name = "Helsinki-NLP/opus-mt-en-fr"  
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

# English to Hindi translation
hindi_model_name = "Helsinki-NLP/opus-mt-en-hi" 
hindi_model = MarianMTModel.from_pretrained(hindi_model_name)
hindi_tokenizer = MarianTokenizer.from_pretrained(hindi_model_name)

# English to German translation
german_model_name = "Helsinki-NLP/opus-mt-en-de" 
german_model = MarianMTModel.from_pretrained(german_model_name)
german_tokenizer = MarianTokenizer.from_pretrained(german_model_name)

# English to Spanish translation
spanish_model_name = "Helsinki-NLP/opus-mt-en-es"  
spanish_model = MarianMTModel.from_pretrained(spanish_model_name)
spanish_tokenizer = MarianTokenizer.from_pretrained(spanish_model_name)

# English to Italian translation
italian_model_name = "Helsinki-NLP/opus-mt-en-it"  
italian_model = MarianMTModel.from_pretrained(italian_model_name)
italian_tokenizer = MarianTokenizer.from_pretrained(italian_model_name)

# English to Russian translation
russian_model_name = "Helsinki-NLP/opus-mt-en-ru"  
russian_model = MarianMTModel.from_pretrained(russian_model_name)
russian_tokenizer = MarianTokenizer.from_pretrained(russian_model_name)

# English to Portuguese translation

# English to Dutch translation
dutch_model_name = "Helsinki-NLP/opus-mt-en-nl"  
dutch_model = MarianMTModel.from_pretrained(dutch_model_name)
dutch_tokenizer = MarianTokenizer.from_pretrained(dutch_model_name)

# English to Chinese translation
chinese_model_name = "Helsinki-NLP/opus-mt-en-zh"  
chinese_model = MarianMTModel.from_pretrained(chinese_model_name)
chinese_tokenizer = MarianTokenizer.from_pretrained(chinese_model_name)

# English to Japanese translation
japanese_model_name = "Helsinki-NLP/opus-mt-en-jap"  
japanese_model = MarianMTModel.from_pretrained(japanese_model_name)
japanese_tokenizer = MarianTokenizer.from_pretrained(japanese_model_name)

# English to Arabic translation
arabic_model_name = "Helsinki-NLP/opus-mt-en-ar" 
arabic_model = MarianMTModel.from_pretrained(arabic_model_name)
arabic_tokenizer = MarianTokenizer.from_pretrained(arabic_model_name)

# Function for translating the summary to the target language

# Define a class for translation
class TextTranslator:
    def __init__(self, target_language):

        self.target_language = target_language
        self.model_name = f"Helsinki-NLP/opus-mt-en-{target_language}"
        self.model = MarianMTModel.from_pretrained(self.model_name)
        self.tokenizer = MarianTokenizer.from_pretrained(self.model_name)

    def translate_text(self, text, target_language):
        if target_language == 'hindi':
            model = hindi_model
            tokenizer = hindi_tokenizer
        elif target_language == 'german':
            model = german_model
            tokenizer = german_tokenizer
        elif target_language == 'spanish':
            model = spanish_model
            tokenizer = spanish_tokenizer
        elif target_language == 'italian':
            model = italian_model
            tokenizer = italian_tokenizer
        elif target_language == 'russian':
            model = russian_model
            tokenizer = russian_tokenizer
        elif target_language == 'dutch':
            model = dutch_model
            tokenizer = dutch_tokenizer
        elif target_language == 'chinese':
            model = chinese_model
            tokenizer = chinese_tokenizer
        elif target_language == 'japanese':
            model = japanese_model
            tokenizer = japanese_tokenizer
        elif target_language == 'arabic':
            model = arabic_model
            tokenizer = arabic_tokenizer
        else:
            # Add more languages as needed
            pass

        inputs = self.tokenizer.encode(text, return_tensors="pt")
        translator = self.model.generate(inputs, max_length=len(text))
        translation = self.tokenizer.decode(translator[0], skip_special_tokens=True)
        return translation

# Define a class for note-making
class NoteMaker:
    @staticmethod
    @st.cache_data
    def generate_note_making(summary_text):
        note_making_rules = """
         Note-Making Rules:
        1. Use a Consistent Format
        2. Capture Key Information
        3. Abbreviations and Symbols
        4. Organize with Headings and Subheadings
        5. Highlight or Use Colors
        6. Be Selective
        7. Active Listening/Reading
        8. Use Lists and Bullets
        9. Whitespace
        10. Date and Page Numbers
        11. Draw Diagrams and Visuals
        12. Summarize and Review
        13. Use Sticky Notes or Flags
        14. Digital Tags or Keywords
        15. Personalization
        16. Stay Organized
        17. Regular Maintenance
        18. Consistency
        19. Share and Collaborate
        20. Practice and Adapt
        """
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a note-making assistant."},
                {"role": "user", "content": summary_text},
                {"role": "assistant", "content": note_making_rules}
            ],
            max_tokens=150,
            temperature=0.7,
            stop=None
        )
        return response.choices[0].message["content"].strip()

# Create a Streamlit web application
class YouTubeTranscriptSummarizerApp:
    def __init__(self):
        self.yt_video = st.text_input("Enter YouTube Video URL: :globe_with_meridians:")
        self.cleaned_text = None

    def run(self):
        st.title("Youtube Transcript Summarizer :100:")

        with st.container():
            col01, col02 = st.columns([2, 2])

            with col01:
                st.subheader("Video :video_camera:")
                if self.yt_video:
                    st.video(self.yt_video)

            with col02:
                st.subheader("Transcript from Video :spiral_note_pad:")
                try:
                    transcript = YouTubeTranscriptApi.get_transcript(VideoIDExtractor.get_video_id(self.yt_video))
                    result = ""
                    for i in transcript:
                        result += ' ' + i['text']
                    self.cleaned_text = re.sub(r'[^A-Za-z\s]+', '', result)
                    st.text_area("Closed Captions", self.cleaned_text, height=380)
                except Exception as e:
                    st.error("An error occurred. Please provide a valid YouTube Video URL.")

        st.write("___________________________________________________________________________________________")

        with st.container():
            col11, col12 = st.columns(2)
            with col11:
                if st.button("Summarize 🤏", key="summarize_button"):
                    st.write("Summarizing...")
                    text = self.cleaned_text
                    cleaned_summary = TextSummarizer.summarize_text(text)
                    st.subheader("Summerized Text (Original):")
                    st.text_area('summary', cleaned_summary, height=321)
            with col12:
                st_lottie(summary, speed=1, key=None)

        st.write("___________________________________________________________________________________________")

        with st.container():
            col21, col22 = st.columns(2)
            with col21:
                st_lottie(trans, speed=1, key=None)
            with col22:
                selected_language = st.selectbox("Select Language for Translation", ['hi','de','es','it','ru','nl','zh','jap','ar','fr'])  # Add more languages as needed
                if st.button("Translate 🗣️🧏‍♀️", key="translate_button"):
                    st.write(f"Translating... to ({selected_language})")
                    text = self.cleaned_text
                    cleaned_summary = TextSummarizer.summarize_text(text)
                    translator = TextTranslator(selected_language)
                    translated_summary = translator.translate_text(cleaned_summary, selected_language)
                    st.text_area('translated', translated_summary, height=280)

        st.write("___________________________________________________________________________________________")

        with st.container():
            col31, col32 = st.columns(2)
            with col31:
                if st.button("Transform 📝", key="transform_button"):
                    st.write("Transforming...")
                    text = self.cleaned_text
                    cleaned_summary = TextSummarizer.summarize_text(text)
                    note_making = NoteMaker.generate_note_making(cleaned_summary)
                    st.subheader("Note-Making:")
                    st.text_area('Note-Making', note_making, height=321)
            with col32:
                st_lottie(note, speed=1, key=None)

if __name__ == "__main__":
    app = YouTubeTranscriptSummarizerApp()
    app.run()
