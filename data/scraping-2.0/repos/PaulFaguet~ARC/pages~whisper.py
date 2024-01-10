import openai
import streamlit as st
import os
import nltk
# from moviepy.editor import *
from dotenv import load_dotenv

load_dotenv()

nltk.download('punkt')

st.set_page_config(layout="wide")

st.title('Transcription Audio → Texte')

# path_to_file = st.secrets["AUDIO_FILE_PATH_MP3"]
path_to_file = os.getenv("AUDIO_FILE_PATH_MP3")

audio_input = st.file_uploader(
    label="Importer un ou plusieurs fichiers mp3", 
    help="Fichiers limités à 25MB avec le type mp3.", 
    accept_multiple_files=True, 
    type=['mp3']
)

formate_checkbox = st.checkbox("Formater le texte sous forme de dialogue", value=False)

if audio_input is not None and st.button('Transcrire'): 
    i = 1
    
    for audio_file in audio_input:
        st.info("Transcription du fichier audio n°%s : %s" % (i, audio_file.name))
        audio_file_path = path_to_file
        with open(audio_file_path, "wb") as f:
            f.write(audio_file.getbuffer())
            transcript = openai.Audio.transcribe("whisper-1", open(audio_file_path, 'rb'))["text"]
            
            if formate_checkbox:
                tokens = nltk.tokenize.word_tokenize(transcript)
                st.info("Formatage du texte sous forme de dialogue.")
                if len(tokens) > 1000:
                    # create a loop to call the API for each part, the first loop take the 1000 first tokens, the second loop take the 1000 next tokens, until the end of the text
                    for i in range(0, len(tokens), 1000):
                        # resp = openai.Completion.create(
                        #     model="text-davinci-003",
                        #     prompt=f"""\
                        #     Transforme le texte ci-dessous sous forme de dialogue. Pour information, il y a deux interlocuteurs. Ne change pas le texte, réécrit le comme il est, juste sous forme de dialogue : revient à la ligne quand l'interlocuteur change.
                        #     L'un des interlocuteurs se nomme Koralyne et pose des questions. L'autre se nomme SIR et lui répond.add()
                            
                        #     Texte à transformer : "{nltk.tokenize.treebank.TreebankWordDetokenizer().detokenize(tokens[i:i+1000])}"
                        #     """,
                        #     max_tokens=1000,
                        #     temperature=0.1,           
                        # )
                        resp = openai.ChatCompletion.create(
                            model="gpt-4-1106-preview",
                            messages=[
                                {"role": "system", "content": "Tu es un expert SEO."},
                                {"role": "user", "content": f"""Transforme le texte ci-dessous sous forme de dialogue. Pour information, il y a deux interlocuteurs. Ne change pas le texte, réécrit le comme il est, juste sous forme de dialogue : revient à la ligne quand l'interlocuteur change.
                                    L'un des interlocuteurs se nomme Koralyne et pose des questions. L'autre se nomme SIR et lui répond.add()
                                    
                                    Texte à transformer : "{nltk.tokenize.treebank.TreebankWordDetokenizer().detokenize(tokens[i:i+1000])}"
                                    """,
                                },
                            ]
                        )
                        st.write(resp["choices"][0]["text"])
                
                else:
                    # resp = openai.Completion.create(
                    #     model="text-davinci-003",
                    #     prompt=f"""\
                    #     Transforme le texte ci-dessous sous forme de dialogue. Pour information, il y a deux interlocuteurs. Ne change pas le texte, réécrit le comme il est, juste sous forme de dialogue : revient à la ligne quand l'interlocuteur change.
                    #     L'un des interlocuteurs se nomme Koralyne et pose des questions. L'autre se nomme SIR et lui répond.add()
                        
                    #     Texte à transformer : "{transcript}"
                    #     """,
                    #     max_tokens=1000,
                    #     temperature=0.1,           
                    # )
                    resp = openai.ChatCompletion.create(
                        model="gpt-4-1106-preview",
                        messages=[
                            {"role": "system", "content": "Tu es un expert SEO."},
                            {"role": "user", "content": f"""Transforme le texte ci-dessous sous forme de dialogue. Pour information, il y a deux interlocuteurs. Ne change pas le texte, réécrit le comme il est, juste sous forme de dialogue : revient à la ligne quand l'interlocuteur change.
                                L'un des interlocuteurs se nomme Koralyne et pose des questions. L'autre se nomme SIR et lui répond.add()
                                
                                Texte à transformer : "{transcript}"
                                """
                            },
                        ]
                    )
                    st.write(resp["choices"][0]["text"])
            else:
                st.write(transcript)
            
            # delete the file
        os.remove(audio_file_path)
        i += 1
    
    st.success("La transcription a été réalisée avec succès.")
