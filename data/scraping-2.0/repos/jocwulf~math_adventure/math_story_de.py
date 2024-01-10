import os
import numpy as np
import streamlit as st
import openai
from dotenv import load_dotenv
from streamlit_chat import message
from PIL import Image

# Konstanten
SENTENCES_PER_EPISODE = 5   # Anzahl der Sätze pro Episode
RIDDLE_MIN = 2 # Mindestzahl für Rätsel
MODEL = "gpt-3.5-turbo-0613" # Verwenden Sie ein besser performantes GPT-Modell, wenn möglich

# Laden des OpenAI-Schlüssels aus der .env-Datei
load_dotenv(".env")
openai.api_key = os.getenv("OPENAI_KEY")

# Bild laden
image = Image.open('children.png')

def reset_state():
    """Setzt den Sitzungsstatus zurück."""
    keys = list(st.session_state.keys())
    for key in keys:
        del st.session_state[key]

def generate_riddle(calculation_type, riddle_max):
    """Generiert ein Rätsel und gibt die Frage und Antwort zurück."""
    num1 = np.random.randint(RIDDLE_MIN, riddle_max)
    num2 = np.random.randint(RIDDLE_MIN, riddle_max)
    if calculation_type == "Addition":
        question = "{} + {}".format(num1, num2)
        answer = num1 + num2
    elif calculation_type == "Subtraction":
        question = "{} - {}".format(max(num1, num2), min(num1, num2))
        answer = max(num1, num2) - min(num1, num2)
    elif calculation_type == "Multiplication":
        question = "{} * {}".format(num1, num2)
        answer = num1 * num2
    elif calculation_type == "Division":
        product = num1 * num2
        question = "{} / {}".format(product, num1)
        answer = num2
    return question, answer

def generate_story(messages):
    """Generiert eine Story-Episode mit der OpenAI-API und gibt die Story zurück."""
    story = openai.ChatCompletion.create(
        model=MODEL,
        messages=messages,
        temperature=0.5,
    )
    return story['choices'][0]['message']['content']

def generate_challenge():
    """Generiert eine Reihe von Rätseln und Geschichten für das Kind zum Lösen."""
    st.session_state['right_answer'] = [] # Liste der richtigen Antworten
    st.session_state['question'] = [] # Liste der Fragen
    st.session_state['story'] = [] # Liste der Episoden  
    st.session_state['num_riddles'] = st.session_state['riddle_count'] # Anzahl der Rätsel, die im Sitzungsstatus bestehen bleiben
    messages = [] # Liste der Nachrichten für OpenAI-Anfragen

    # Systemnachricht für OpenAI-Anfragen
    sys_message = """Erzähle einem siebenjährigen Kind eine Fortsetzungsgeschichte über {2}. Jede Episode besteht genau aus {0} Sätzen. Die Geschichte handelt vom Tag von {1}. Eine Episode der Geschichte besteht genau aus {0} Sätzen und nicht mehr. Beginne direkt mit der Erzählung. Beende jede Episode mit einem Matheproblem, das immer zuvor von [role: user] gestellt wird. Integriere das Matheproblem in die Erzählung der Episode. Stelle sicher, dass das Matheproblem korrekt formuliert ist. Gib die Lösung nicht. Durch das Lösen dieses Problems kann das Kind {1} helfen. Führe in der neuen Episode bereits erzählte Episoden fort und stelle ein neues Matheproblem. BITTE BEACHTEN: Gib die Lösung zum Matheproblem nicht. Verwende nur {0} Sätze. Beende das Ende mit dem Matheproblem.""".format(SENTENCES_PER_EPISODE, st.session_state['person'], st.session_state['topic'])
    
    messages.append({"role": "system", "content": sys_message})

    # Fortschrittsbalken erstellen
    progress_bar = st.progress(0)
    status_message = st.empty()
    status_message.text("Ich generiere deine Geschichte...")
    for i in range(st.session_state['riddle_count']): # Rätsel und Geschichten generieren
        # Fortschrittsbalken aktualisieren
        progress_bar.progress((i + 1) / st.session_state['riddle_count'])

        # Rätsel generieren
        calculation_type = np.random.choice(st.session_state['calculation_type'])
        question, answer = generate_riddle(calculation_type, st.session_state['riddle_max'])
        messages.append({"role": "user", "content": question})

        # Geschichte generieren
        story = generate_story(messages)
        messages.append({"role": "assistant", "content": story})
        
        # Rätsel und Geschichte im Sitzungsstatus speichern
        st.session_state.right_answer.append(answer)
        st.session_state.question.append(question)
        st.session_state.story.append(story)

    # Finale Episode erstellen
    messages.pop(0) # Erstes Element in der Nachrichtenliste entfernen (Systemnachricht)
    messages.append({"role": "user", "content": "Beende die Geschichte in fünf Sätzen. Füge kein Matheproblem hinzu."})
    story = generate_story(messages)
    st.session_state.story.append(story)   
    
    st.session_state['current_task'] = 0 # Verfolgt die aktuelle Episode
    status_message.empty()  # Statusnachricht entfernen
    return st.session_state['story'][0] # Gibt die erste Episode zurück

def on_input_change():
    """Verarbeitet die Eingabe des Kindes und überprüft, ob sie korrekt ist."""
    user_input = st.session_state["user_input"+str(st.session_state['current_task'])] # Benutzereingabe holen
    st.session_state['past'].append(user_input) # Benutzereingabe im Sitzungsstatus speichern
    if user_input == st.session_state.right_answer[st.session_state['current_task']]: # Benutzereingabe ist korrekt
        # Überprüfen, ob alle Aufgaben erledigt sind
        if st.session_state['current_task'] == st.session_state['num_riddles']-1: # Alle Aufgaben sind erledigt
            st.session_state['generated'].append(st.session_state['story'][st.session_state['current_task']+1]) # Finale Episode generieren
            st.session_state['finished'] = True # Fertig-Flag setzen
        else: # Nicht alle Aufgaben sind erledigt
            st.session_state['current_task']+=1 # Aufgabenzähler erhöhen
            st.session_state['generated'].append(st.session_state['story'][st.session_state['current_task']]) # Nächste Episode für die Ausgabe anhängen
    else: # Benutzereingabe ist falsch
        st.session_state.generated.append("Nicht ganz richtig. Versuche es noch einmal! " + st.session_state.question[st.session_state['current_task'] ])  # Falsche Nachricht zur Ausgabe hinzufügen

if 'end_story' not in st.session_state:
    st.session_state['end_story'] = False

if 'input_done' not in st.session_state:
    st.session_state['input_done'] = False

st.title("༼ ͡ಠ ͜ʖ ͡ಠ ༽ Dein Mathe-Abenteuer")
st.image(image, use_column_width=True, caption = "3 * 2 = 7 ???")

if st.session_state['end_story']: # Geschichte beendet
    st.write("Die Geschichte ist zu Ende.")
    if st.button("Eine neue Geschichte starten"):
        reset_state()
else: # Geschichte nicht beendet
    if st.session_state['input_done'] == False:
        with st.sidebar:
            st.selectbox("Wie viele Matheaufgaben möchtest du lösen?", [3, 5, 7, 10], key="riddle_count", index=0)
            st.multiselect("Wähle den Rechentyp", ["Addition", "Subtraktion", "Multiplikation", "Division"], key="calculation_type", default=["Addition", "Subtraktion", "Multiplikation", "Division"])
            st.selectbox("Wähle den Zahlenbereich", ["1 Stelle (1-9)", "2 Stellen (1-99)"], key="number_range", index=0)
            st.text_input("Wer soll die Hauptfigur Deiner Geschichte sein?", key="person")
            st.text_input("Was willst Du mit Deiner Hauptfigur erleben?", key="topic")
            if st.button("Die Geschichte starten", key="start_btn"):
                st.session_state['input_done'] = True
                if st.session_state['number_range'] == "1 Stelle (1-9)":
                    st.session_state['riddle_max'] = 9
                else:
                    st.session_state['riddle_max'] = 99

    if st.session_state['input_done']:
        if 'past' not in st.session_state:
            st.session_state['past']=['Hier werden deine Antworten angezeigt.']
        if 'generated' not in st.session_state:
            st.session_state['generated'] = [generate_challenge()]
        if 'finished' not in st.session_state:
            st.session_state['finished'] = False
        chat_placeholder = st.empty()
        with chat_placeholder.container():    
            # st.write(st.session_state.story) # zum Debuggen
            for i in range(len(st.session_state['generated'])):  
                            
                message(str(st.session_state['past'][i]), is_user=True, key=str(i) + '_user')
                message(
                    st.session_state['generated'][i],
                    key=str(i)
                )

        if not st.session_state['finished']:
            with st.container():
                st.number_input("Deine Lösung:", min_value=-1, max_value=100, 
                                value=-1, step=1, on_change=on_input_change, 
                                key="user_input"+str(st.session_state['current_task']))
        if st.button("Die Geschichte beenden", key="end_btn"):
            st.session_state['end_story'] = True
