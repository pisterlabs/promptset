import time
import streamlit as st
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
import get_prompt
from langchain.schema import AIMessage, HumanMessage
from StreamHandler import StreamHandler


def handle_messages():
    """Gestisce i messaggi della chat.

    Inizializza lo stato della sessione. Se "messages" non è presente in st.session_state,
    lo inizializza a una lista vuota. Successivamente, gestisce i messaggi presenti in
    st.session_state["messages"], scrivendo i messaggi degli utenti e dell'assistente
    nella chat.

    """

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for msg in st.session_state["messages"]:
        if isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        else:
            st.chat_message("assistant").write(msg.content)


def display_lesson(lesson_selection, lesson_info):
    """Visualizza una lezione specifica.

    Parameters:
    lesson_selection (str): Il titolo della lezione da visualizzare.
    lesson_info (dict): Un dizionario contenente le informazioni sulla lezione, con la chiave "description" per la descrizione.

    Returns:
    None

    """

    with st.container():
        st.markdown(f"**{lesson_selection}**")
        st.write(lesson_info["description"])


def run_langchain_model(prompt, lesson_type, lesson_content, lesson_selection, openai_api_key):
    """Esegue il modello Langchain per gestire le lezioni e interagire con l'utente tramite il chatbot.

    Parameters:
    prompt (str): Il prompt iniziale per il modello.
    lesson_type (str): Il tipo di lezione.
    lesson_content (str): Il contenuto della lezione.
    lesson_selection (str): La selezione della lezione.
    openai_api_key (str): La chiave API di OpenAI per l'accesso al modello.

    """

    try:

        # Set up a streaming handler for the model
        with st.chat_message("assistant"):
            stream_handler = StreamHandler(st.empty())
            model = ChatOpenAI(streaming=True, callbacks=[stream_handler], model="gpt-3.5-turbo-16k",
                               openai_api_key=openai_api_key)

            # Load a prompt template based on the lesson type
            if lesson_type == "Instructions based lesson":
                prompt_template = get_prompt.load_prompt(content=lesson_content)
            else:
                prompt_template = get_prompt.load_prompt_with_questions(content=lesson_content)

            # Run a chain of the prompt and the language model
            chain = LLMChain(prompt=prompt_template, llm=model)
            response = chain(
                {"input": prompt, "chat_history": st.session_state.messages[-20:]},
                include_run_info=True,
                tags=[lesson_selection, lesson_type]
            )
            st.session_state.messages.append(HumanMessage(content=prompt))
            st.session_state.messages.append(AIMessage(content=response[chain.output_key]))

    except Exception as e:
        # Handle any errors that occur during the execution of the code
        st.error(f"An error occurred: {e}")


@st.cache_data()
def get_lesson_content(lesson_file):
    """Ottiene il contenuto di una lezione da un file.

    Parameters:
    lesson_file (str): Il percorso del file della lezione.

    Returns:
    str: Il contenuto della lezione.

    """

    try:
        with open(lesson_file, "r") as f:
            return f.read()
    except FileNotFoundError:
        st.error(f"Error: Lesson file not found at {lesson_file}")
        st.stop()


def download_chat():
    """Genera e scarica la conversazione nel formato HTML.

    La funzione genera un file HTML che rappresenta la conversazione
    registrata tra l'utente e l'assistente. Il file HTML include
    messaggi dell'utente e dell'assistente formattati.

    """

    messages = st.session_state.get("messages", [])  # Retrieve messages from session state

    chat_content = "<html><head><link rel='stylesheet' type='text/css' href='styles.css'></head><body>"
    for msg in messages:
        if isinstance(msg, AIMessage):
            chat_content += f"<p class='message ai-message'><strong>AI:</strong> {msg.content}</p>"
        elif isinstance(msg, HumanMessage):
            chat_content += f"<p class='message user-message'><strong>User:</strong> {msg.content}</p>"
        else:
            chat_content += f"<p class='message'>Unknown Message Type: {msg}</p>"

    chat_content += "</body></html>"

    with open("chat.html", "w", encoding="utf-8") as html_file:
        html_file.write(chat_content)

    # Download the generated HTML file
    st.download_button("Download Chat", open("chat.html", "rb"), key="download_chat", file_name="chat.html",
                       mime="text/html")


def reset_lesson():
    """Ripristina lo stato della lezione.

    La funzione reimposta diversi attributi nello stato della sessione a valori vuoti o None,
    consentendo di ripartire da zero in una nuova lezione.

    """

    st.session_state["messages"] = []
    st.session_state["completed_lessons"] = []
    st.session_state["current_lesson"] = None
    st.session_state["current_lesson_type"] = None
    st.session_state["code_snippet"] = None


def setup_page():
    """Configura la pagina per l'applicazione.

    Questa funzione configura la pagina dell'applicazione, impostando il titolo e l'icona.

    """

    st.set_page_config(page_title="AIDE", page_icon="🤖")
    st.title("AIDE: Studiare non è mai stato così facile! Aide è qui per guidarti!")


def avanzamento_barra(connection):
    """Gestisce la barra di avanzamento e il punteggio associato ai messaggi.

    La funzione controlla i messaggi presenti nello stato della sessione e aggiorna una barra di avanzamento
    nel sidebar in base al numero di messaggi di risposta corretta.

    """

    # inizializzazione variabili
    bar = st.progress(0)
    bar.empty()
    contatore = 0

    cursor = connection.cursor()
    query = "SELECT COUNT(*) FROM Lezioni"

    cursor.execute(query)
    result = cursor.fetchall()

    messages = st.session_state.get("messages", [])
    for msg in messages:
        if isinstance(msg, AIMessage):
            if msg.content.startswith("Hai risposto correttamente!") or msg.content.startswith("That's correct!"):
                contatore += 1
    num_lezioni = 100 / result[0][0]
    progresso = contatore * num_lezioni
    bar = st.sidebar.progress(progresso, "Punteggio")
    time.sleep(1)


def load_lesson_content(lesson_file):
    """Carica il contenuto di una lezione da un file.

    Parameters:
    lesson_file (str): Il percorso del file della lezione.

    Returns:
    str: Il contenuto della lezione.

    Raises:
    FileNotFoundError: Se il file della lezione non è trovato.

    """

    try:
        with open(lesson_file, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        st.error(f"Error: Lesson file not found at {lesson_file}")
        st.stop()
