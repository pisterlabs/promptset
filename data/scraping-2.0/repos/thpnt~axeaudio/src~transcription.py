def process_audio(status_label, root) :
    from config import DATA_DIRECTORY, TRANSCRIPT_DIRECTORY, EDITED_DIRECTORY
    from prompt import prompt_article, prompt_cdc
    from utils import get_most_recent_file, store_transcript, get_completion
    from dotenv import load_dotenv
    import openai
    import os
    from tkinter import simpledialog
    import time


    #OpenAI creds
    _  = load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    openai.organizations = os.getenv("OPENAI_ORGANISATION_KEY")

        
    status_label.config(text = "Transcribing text with Whisper ...")
    root.update()
    #Get the audio most recent_audio files in the data_directory
    #cwd = os.getcwd()
    #PATH = os.path.abspath(os.path.join(cwd, os.pardir))
    PATH = os.getcwd()
    file_name = get_most_recent_file(DATA_DIRECTORY)
    audio_file = open(f"{PATH}/{file_name}", 'rb')


    #Transcript audio_file
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    text = transcript['text']
    #Store raw_transcript
    status_label.config(text = "Storing raw transcripts in data/raw_transcripts/ ...")
    root.update()
    store_transcript(text, TRANSCRIPT_DIRECTORY, DATA_DIRECTORY)


    #Text transformation with GPT
    #treatment_type = input('You want to transcribe a web articles ? ("y" if True else "n" then type enter) : ')
    treatment_type = simpledialog.askstring("Input", "Enter <y> for web article treatment, <n> for CDC treatment:")

    while not any(treatment_type == x for x in ['y', 'n']) :
        #treatment_type = input('Invalid input. Enter <y> for web article treatment, <n> for CDC treatment : ')
        treatment_type = simpledialog.askstring("Input", 'Invalid input. Enter <y> for web article treatment, <n> for CDC treatment : ')

    status_label.config(text = "Editing text with GPT ...")
    root.update()
    match treatment_type:
        case 'y' :
            prompt = prompt_article(text)
            response = get_completion(prompt)
        case 'n' :
            prompt = prompt_cdc(text)
            response = get_completion(prompt)

    status_label.config(text = "Storing edited text in data/edited_transcripts/ ...")
    root.update()
    store_transcript(response, EDITED_DIRECTORY, DATA_DIRECTORY)
    status_label.config(text = 'Edited text is stored in data/edited_transcripts.')
    root.update()
    time.sleep(3)
    status_label.config(text = '')
    root.update()
    return

        
