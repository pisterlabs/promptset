import streamlit as st
import genanki
import random
from os.path import exists
import time
from gtts import gTTS
import os
from gen_audio import TextToSpeechService
from gen_pdf import PDFGenerator
from audio_recorder_streamlit import audio_recorder
import openai
# from models.anki_model import gen_anki_id, my_model
from controllers.ai_transcriber import AiTrancriber

# make the api key a text input that is hidden
# if there is a file called openai_api_key.txt, use the api key in that file
if "open_ai_api_key.txt" in os.listdir():
    with open("open_ai_api_key.txt", "r") as f:
        api_key = f.read()
else:
    api_key = st.text_input("OpenAI API key", type="password")


openai.api_key = api_key

model_engine = "text-davinci-003"

SEPARATOR = ";"

# create a list of languages that can be used with the google text to speech API
google_language_list = ['af', 'sq', 'ar', 'hy', 'bn', 'ca', 'zh', 'zh-cn', 'zh-tw', 'zh-yue', 'hr', 'cs', 'da', 'nl', 'en', 'en-au', 'en-uk', 'en-us', 'eo', 'fi', 'fr', 'de', 'el', 'hi', 'hu', 'is', 'id', 'it', 'ja', 'ko', 'la', 'lv', 'mk', 'no', 'pl', 'pt', 'pt-br', 'ro', 'ru', 'sr', 'sk', 'es', 'es-es', 'es-us', 'sw', 'sv', 'ta', 'th', 'tr', 'vi', 'cy']

language_list_deepl = ["BG","CS","DA","DE","EL","EN-GB","EN-US","ES","ET","FI","FR","HU","ID","IT","JA","KO","LT","LV","NB","NL","PL","PT-BR","PT-PT","RO","RU","SK","SL","SV","TR","UK","ZH"]

def gen_anki_id():
    return random.randrange(1 << 30, 1 << 31)


# add an upload button to upload an audio file
audio_upload = st.file_uploader("Upload an audio file", type=["mp3"], accept_multiple_files=False)
# rename the audio file to audio.mp3
if audio_upload:
    with open("audio.mp3", "wb") as f:
        f.write(audio_upload.read())

ai_generated_phrases = None
target_language = st.selectbox("Target language", language_list_deepl, index=6)

# the following line is used to create a button that will record audio
audio_bytes = audio_recorder()

# the following if statement
if audio_bytes:
    with open("audio.mp3", "wb") as f:
        f.write(audio_bytes)
# add a button to delete the auddio file
if st.button("Delete audio file"):
    if "audio.mp3" in os.listdir():
        os.remove("audio.mp3")
    if audio_bytes:
        audio_bytes = None

if audio_bytes or audio_upload:
    # use the lastest audio file
    # convert audio_bytes to audio.mp3
    audio_file = audio_bytes or audio_upload
    # audio_upload if audio_upload else audio_bytes
    st.audio(audio_file, format="audio/mp3")
    if "audio.mp3" in os.listdir():
        at = AiTrancriber(audio_file, target_language)
        ai_generated_phrases = at.format_phrases()


def txt_to_list_of_lists(text_input):
    table = [line.split(SEPARATOR) for line in text_input.split("\n")]
    # remove any empty lines
    table = [line for line in table if line != ['']]
    return table


def exists(file_name):
    try:
        with open(file_name):
            return True
    except IOError:
        return False


# create a function that takes a string and a language value and returns an audio file using the google text to speech API
def text_to_audio(text, language):
    # create a string that will be used to create the audio file name
    audio_file_name = text.replace(" ", "_") + ".mp3"
    # create a string that will be used to create the audio file path
    audio_file_path = f"{audio_file_name}"
    # if the audio file does not exist then create it
    if not exists(audio_file_path):
        # create an audio file using the google text to speech API
        tts = gTTS(text, lang=language)
        tts.save(audio_file_path)
    return audio_file_path




def to_anki_deck_no_audio(table, language_one, language_two, deck_title):
    my_deck = genanki.Deck(gen_anki_id(), deck_title)
    package = genanki.Package(my_deck) 
    i = 0
    my_bar = st.progress(i)
    for row in table:
        my_bar.progress(i/len(table))
        try:
            note = genanki.Note(
                    model = my_model,
                    fields = [row[0], row[1]],                    
                )
        except IndexError:
            pass
        my_deck.add_note(note)
    package.write_to_file(f"{deck_title}.apkg")
    return f"{deck_title}.apkg"


def to_anki_deck_with_audio(table, language_one, language_two, deck_title):
    # similar to the function above but with audio
    # for each row in the table, create an audio file for each language using the text_to_audio function
    # then add the audio file to the note
    my_deck = genanki.Deck(gen_anki_id(), deck_title)
    package = genanki.Package(my_deck)
    i = 0
    my_bar = st.progress(i)
    for row in table:
        # create the audio file for language one
        audio_file_one = text_to_audio(row[0], language_one)
        audio_one = '[sound:{}]'.format(audio_file_one)
        print(audio_file_one)
        # create the audio file for language two
        audio_file_two = text_to_audio(row[1], language_two)
        audio_two = '[sound:{}]'.format(audio_file_two)
        # add the audio files to the note
        my_bar.progress(i/len(table))
        try:
            note = genanki.Note(
                    model = my_model,
                    fields = [row[0], row[1], audio_one, audio_two]
                )
        except IndexError:
            pass
        package.media_files.append(audio_file_one)
        package.media_files.append(audio_file_two)
        my_deck.add_note(note)
    package.write_to_file(f"{deck_title}.apkg")
    return f"{deck_title}.apkg"

def cleanup():
    # delete all the audio files
    print("removing audio files and apkg files")
    for file in os.listdir():
        if file.endswith(".mp3"):
            os.remove(file)
        elif file.endswith(".apkg"):
            os.remove(file)
            

def main():    
    anki_deck = None
    audio_lesson = None
    pdf_lesson = None
    st.title("Create your Lesson below!")
    with st.form("input phrases"):
    # pick a target language

        
        # create 2 input fields called language one and language two for inputing the language as a string (5 characters max)
        # use a selector to select the language instead of a text input
        language_one = st.selectbox("Language One", google_language_list, index=14)        
        language_two = st.selectbox("Language Two", google_language_list, index=20)
        # create a toggle button to select if you want to use audio or not
        use_audio = st.checkbox("Use Audio")
        st.text("This is a simple text input, use ; to separate the fields")
        # create a text area for inputing the phrases with default text """bonjour ; hello
        # merci ; thank you"""
        if ai_generated_phrases:
            # prefill the form with the generated phrases
            content = st.text_area(label= "input your vocabulary" , value=ai_generated_phrases ,placeholder=ai_generated_phrases, height=200)
        else:            
            content = st.text_area(label= "input your vocabulary" ,placeholder="""hello ; bonjour
thank you ; merci""", height=200)
        intro_outro_language = st.selectbox("Intro/Outro Language", google_language_list, index=14)
        intro = st.text_area(label= "input your intro" ,placeholder="Lesson 1", height=20)
        outro = st.text_area(label= "input your outro" ,placeholder="Thank you for listening", height=20)
        table = txt_to_list_of_lists(content)
        deck_title = st.text_input(label="Deck Title", value="My Lesson")
        submit_anki = st.form_submit_button("Generate Anki Deck")
        silence_between_blocks_duration = st.number_input("Silence between blocks duration", min_value=0, max_value=10, value=1)
        silence_between_phrases_duration = st.number_input("Silence between phrases duration", min_value=0, max_value=10, value=2)
        block_type = st.selectbox("Block Type", ["recall", "repeat", "recognize"])
        shuffle_blocks = st.checkbox("Shuffle Blocks")
        submit_audio = st.form_submit_button("Generate Audio")
        submit_pdf = st.form_submit_button("Generate PDF")
        # st.write(table)
        if submit_anki:
            if use_audio:
                anki_deck = to_anki_deck_with_audio(table, language_one, language_two, deck_title)
            else:
                anki_deck = to_anki_deck_no_audio(table, language_one, language_one, deck_title)   
        if submit_audio:
            ttss = TextToSpeechService(
                text=content,
                language_1=language_one,
                language_2=language_two,
                file_name=deck_title,
                separator=";",
                silence_between_blocks_duration=silence_between_blocks_duration,
                silence_between_phrases_duration=silence_between_phrases_duration,
                shuffle_blocks=shuffle_blocks,
                block_type = block_type,
                intro_text = intro,
                outro_text = outro,
                intro_outro_language = intro_outro_language
                )
            st.text(ttss.intro_text + " " + ttss.outro_text)
            ttss.generate_blocks()
            st.text("number of blocks: " + str(len(ttss.blocks_to_concat)))
            ttss.create_audio_file()
            audio_lesson = ttss.get_audio_file()
        if submit_pdf:
            pdf_lesson = PDFGenerator(
                text=content,
                language_1=language_one,
                language_2=language_two,
                file_name=deck_title
                )
            pdf_lesson.format_content()
    # if a file named f"{deck_title}.apkg" exists, in the current folder then show the link to the file 
    if anki_deck:
        st.text("Your deck has been created")            
        st.text("You can download it from the link below")   
        with open(f"{deck_title}.apkg", 'rb') as f:
            contents = f.read()
            st.download_button("Download my deck", data=contents, file_name=f"{deck_title}.apkg")
            # wait for one minute before deleting the audio files
            time.sleep(60)
            cleanup()
    if audio_lesson:
        st.text("Your audio lesson has been created")            
        st.text("You can download it from the link below")   
        with open(f"{deck_title}.mp3", 'rb') as f:
            contents = f.read()
            st.download_button("Download my audio lesson", data=contents, file_name=f"{deck_title}.mp3")
            # wait for one minute before deleting the audio files
            time.sleep(60)
            ttss.delete_audio_files()
    if pdf_lesson:
        st.text("Your pdf lesson has been created")            
        st.text("You can download it from the link below")   
        with open(f"{deck_title}.pdf", 'rb') as f:
            contents = f.read()
            st.download_button("Download my pdf lesson", data=contents, file_name=f"{deck_title}.pdf")
            # wait for one minute before deleting the audio files
            time.sleep(60)
            pdf_lesson.delete_pdf_file()





if __name__ == '__main__':
    main()