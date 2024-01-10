import streamlit as st
import time

from langchain.schema import AIMessage

st.title("Dashboard dello Studente")
container_centrale = st.container()

if "completed_lessons" in st.session_state:
    st.subheader("Lezioni Svolte:")
    for lesson in st.session_state.completed_lessons:
        st.write(f"- {lesson}")
else:
    st.info("Nessuna leziona completata, se desideri conoscere il tuo progresso clicca 'Show Progress'")


def avanzamento_barra():
    # inizializzazione variabili
    bar = st.progress(0)
    bar.empty()
    contatore = 0

    messages = st.session_state.get("messages", [])
    for msg in messages:
        if isinstance(msg, AIMessage):
            if msg.content.startswith("Hai risposto correttamente!"):
                contatore += 1
    progresso = contatore * 10
    bar = st.progress(progresso, "Punteggio")
    time.sleep(1)


# AVANZAMENTO BARRA PROGRESSO
container_button = st.sidebar.container()
container_button = st.empty()
button = container_button.button("Show Progress", on_click=None)

if button:
    container_button.empty()
    button_hide = container_button.button("Hide Progress", on_click=None)
    container_centrale = avanzamento_barra()

