import openai
import streamlit as st
import os
import pandas as pd
import json

from time import sleep
from azure.storage.blob import BlobServiceClient
# from pdfkit import configuration, from_string
from myfunc.mojafunkcija import (
    positive_login,
    read_aad_username,
    load_data_from_azure,
    upload_data_to_azure,
    inner_hybrid)

import nltk     # kasnije ce se paketi importovati u funkcijama
from langchain.utilities import GoogleSerperAPIWrapper
from st_copy_to_clipboard import st_copy_to_clipboard
from streamlit_extras.stylable_container import stylable_container


st.set_page_config(page_title="Zapisnik", page_icon="🤖")

version = "v1.1.1 Azure, username i upload file"
os.getenv("OPENAI_API_KEY")
client = openai
assistant_id = "asst_cLf9awhvTT1zxY23K3ebpXbs"  # printuje se u drugoj skripti, a moze jelte da se vidi i na OpenAI Playground-u
client.beta.assistants.retrieve(assistant_id=assistant_id)

# isprobati da li ovo radi kod Vas
# from custom_theme import custom_streamlit_style

ovaj_asistent = "zapisnik"


global username


def main():
    if "username" not in st.session_state:
        st.session_state.username = "positive"
    if deployment_environment == "Azure":    
        st.session_state.username = read_aad_username()
    elif deployment_environment == "Windows":
        st.session_state.username = "lokal"
    elif deployment_environment == "Streamlit":
        st.session_state.username = username
    
    with st.sidebar:
        st.info(
            f"Prijavljeni ste kao: {st.session_state.username}")
        
    client = openai.OpenAI()
    if "data" not in st.session_state:
        st.session_state.data = None
    if "blob_service_client" not in st.session_state:
        st.session_state.blob_service_client = BlobServiceClient.from_connection_string(os.environ.get("AZ_BLOB_API_KEY"))
    if "delete_thread_id" not in st.session_state:
        st.session_state.delete_thread_id = None

    st.session_state.data = load_data_from_azure(st.session_state.blob_service_client)

    try:
        st.session_state.data = st.session_state.data[st.session_state.data.ID != st.session_state.delete_thread_id]
    except:
        pass
    
    threads_dict = {thread.chat: thread.ID for thread in st.session_state.data.itertuples() if st.session_state.username == thread.user and ovaj_asistent == thread.assistant and thread.ID is not st.session_state.delete_thread_id}

    # Inicijalizacija session state-a
    default_session_states = {
        "file_id_list": [],
        "openai_model": "gpt-4-1106-preview",
        "messages": [],
        "thread_id": None,
        "is_deleted": False,
        "cancel_run": None,
        "namespace": "zapisnici",
        "columns": ["user", "chat", "ID", "assistant", "fajlovi"],
        }
    for key, value in default_session_states.items():
        if key not in st.session_state:
            st.session_state[key] = value


    def web_serach_process(q: str) -> str:
        return GoogleSerperAPIWrapper(environment=os.environ["SERPER_API_KEY"]).run(q)


    def hybrid_search_process(upit: str) -> str:
        stringic = inner_hybrid(upit)
        return stringic

    # krecemo polako i sa definisanjem UI-a
   
    # st.markdown(custom_streamlit_style, unsafe_allow_html=True)   # ne radi izgleda vise
    st.sidebar.header(body="Zapisnik asistent")
    st.sidebar.caption(f"Ver. {version}")
    
    with st.sidebar.expander(label="Kako koristiti?", expanded= False):
        st.write(""" 
1. Aplikacija vam omogucava da razgovarate o pitanjima vezanim za interna dokumenta, pravilnike i sl. Pomenite sistematizaciju ili pravilnik. 

2. Pamti razgovore koje ste imali do sada i mozete ih nastaviti po zelji. Odaberite iz padajuceg menija raniji razgovor i odaberite select

3. Mozete zapoceti i novi ragovor. unesite ime novog razgovora i pritisnite novi razgovor , a zatim ga iz padajuceg menija odaberite i potvrdite izbor.

4. Mozete uploadovati dokument i razgovarati o njegovom sadrzaju.

5. Ova aplikacija nema pristup internetu, ali poseduje znanja do Marta 2023. godine.

6. Za neke odgovore mozda ce trebati malo vremena, budite strpljivi.
        """)
    # Narednih 50-tak linija su za unosenje raznih vrednosti
    st.sidebar.text("")

    # funkcije za scrape-ovanje i upload-ovanje dokumenata
    def upload_to_openai(filepath):
        with open(filepath, "rb") as f:
            response = openai.files.create(file=f.read(), purpose="assistants")
        return response.id
    
    uploaded_file = st.sidebar.file_uploader(label="Upload fajla u OpenAI embeding", key="uploadedfile")
    if st.sidebar.button(label="Upload File", key="uploadfile"):
        try:
            with open(file=f"{uploaded_file.name}", mode="wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.file_id_list.append(
                upload_to_openai(filepath=f"{uploaded_file.name}"))
            try:
                st.session_state.data = st.session_state.data.drop(st.session_state.data.columns[[5, 6]], axis=1)
            except:
                pass

            st.session_state.data.loc[st.session_state.data["ID"] == st.session_state.thread_id, 
                                      "fajlovi"].apply(lambda g: g.append(st.session_state.file_id_list[-1]) or g)
            upload_data_to_azure(st.session_state.data)
            client.beta.assistants.files.create(assistant_id="asst_289ViiMYpvV4UGn3mRHgOAr4", file_id=st.session_state.file_id_list[-1])
            
        except Exception as e:
            st.warning("Opis greške:\n\n" + str(e))

    st.sidebar.text("")
    new_chat_name = st.sidebar.text_input(label="Unesite ime za novi chat", key="newchatname")
    if new_chat_name.strip() != "" and st.sidebar.button(label="Create Chat", key="createchat"):
        thread = client.beta.threads.create()
        st.session_state.thread_id = thread.id

        new_row = pd.DataFrame([st.session_state.username, new_chat_name, st.session_state.thread_id, ovaj_asistent, "[]"]).T
        new_row.columns = st.session_state.data.columns

        upload_data_to_azure(pd.concat([st.session_state.data, new_row]))
        st.rerun()
    
    chosen_chat = st.sidebar.selectbox(label="Izaberite chat", options=["Select..."] + list(threads_dict.keys()))
    if chosen_chat.strip() not in ["", "Select..."] and st.sidebar.button(label="Select Chat", key="selectchat2"):
        thread = client.beta.threads.retrieve(thread_id=threads_dict.get(chosen_chat))
        st.session_state.thread_id = thread.id
        st.rerun()

    st.sidebar.text("")

    chat_for_deletion = st.sidebar.selectbox(label="Delete chat", options=["Select..."] + list(threads_dict.keys()), key="chatfordeletion")
    if chat_for_deletion.strip() not in ["", "Select..."] and st.sidebar.button(label="Delete Chat", key="deletechat"):
        thread = client.beta.threads.retrieve(thread_id=threads_dict.get(chat_for_deletion))
        files_for_deletion = st.session_state.data.loc[st.session_state.data["ID"] == thread.id, "fajlovi"].values[0]
        for file_id in files_for_deletion:
            client.beta.assistants.files.delete(
                assistant_id=assistant_id,
                file_id=file_id,
            )

        st.session_state.delete_thread_id = thread.id
        st.session_state.is_deleted = True
        st.rerun()

    if st.session_state.is_deleted:
        upload_data_to_azure(st.session_state.data)
        st.session_state.is_deleted = False

    st.sidebar.text("")
    assistant = client.beta.assistants.retrieve(assistant_id=assistant_id)
    if st.session_state.thread_id:
        thread = client.beta.threads.retrieve(thread_id=st.session_state.thread_id)

    instructions = "Please remember to always check each time for every new question if a tool is relevant to your query. \
    Answer only in the Serbian language."

    # ako se desi error run ce po default-u trajati 10 min pre no sto se prekine -- ovo je da ne moramo da cekamo
    try:
        run = client.beta.threads.runs.cancel(thread_id=st.session_state.thread_id, run_id=st.session_state.cancel_run)
    except:
        pass
    run = None


    # pitalica
    if prompt := st.chat_input(placeholder="Postavite pitanje"):
        if st.session_state.thread_id is not None:
            client.beta.threads.messages.create(thread_id=st.session_state.thread_id, role="user", content=prompt) 

            run = client.beta.threads.runs.create(thread_id=st.session_state.thread_id, assistant_id=assistant.id, 
                                                instructions=instructions)
        else:
            st.warning("Molimo Vas da izaberete postojeci ili da kreirate novi chat.")

    with stylable_container(
                    key="bottom_content",
                    css_styles="""
                        {
                            position: fixed;
                            bottom: 150px;
                        }
                        """,
                    ):
        # obrada upita        
        with st.spinner("🤖 Chatbot razmislja..."):
            # ako se poziva neka funkcija
            if run is not None:
                while True:
            
                    sleep(0.3)
                    run_status = client.beta.threads.runs.retrieve(thread_id=st.session_state.thread_id, run_id=run.id)

                    if run_status.status == 'completed':
                        break

                    elif run_status.status == 'requires_action':
                        tools_outputs = []

                        for tool_call in run_status.required_action.submit_tool_outputs.tool_calls:
                            if tool_call.function.name == "web_search_process":
                                arguments = json.loads(tool_call.function.arguments)
                                try:
                                    output = web_serach_process(arguments["query"])
                                except:
                                    output = web_serach_process(arguments["q"])

                                tool_output = {"tool_call_id":tool_call.id, "output": json.dumps(output)}
                                tools_outputs.append(tool_output)
                    
                            elif tool_call.function.name == "web_search_process":
                                arguments = json.loads(tool_call.function.arguments)
                                tool_output = {"tool_call_id":tool_call.id, "output": json.dumps(output)}
                                tools_outputs.append(tool_output)

                            elif tool_call.function.name == "hybrid_search_process":
                                arguments = json.loads(tool_call.function.arguments)
                                output = hybrid_search_process(arguments["upit"])
                                tool_output = {"tool_call_id":tool_call.id, "output": json.dumps(output)}
                                tools_outputs.append(tool_output)

                        if run_status.required_action.type == 'submit_tool_outputs':
                            client.beta.threads.runs.submit_tool_outputs(thread_id=st.session_state.thread_id, run_id=run.id, tool_outputs=tools_outputs)

                        sleep(0.3)

    try:
        messages = client.beta.threads.messages.list(thread_id=st.session_state.thread_id) 
        for msg in reversed(messages.data): 
            role = msg.role
            content = msg.content[0].text.value 
            if role == 'user':
                st.markdown(f"<div style='background-color:lightblue; padding:10px; margin:5px; border-radius:5px;'><span style='color:blue'>👤 {role.capitalize()}:</span> {content}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div style='background-color:lightgray; padding:10px; margin:5px; border-radius:5px;'><span style='color:red'>🤖 {role.capitalize()}:</span> {content}</div>", unsafe_allow_html=True)
                # copy to clipboard dugme (za svaki odgovor)
                st_copy_to_clipboard(content)
    except:
        pass

# Deployment on Stremalit Login functionality
deployment_environment = os.environ.get("DEPLOYMENT_ENVIRONMENT")

if deployment_environment == "Streamlit":
    name, authentication_status, username = positive_login(main, " ")
else:
    if __name__ == "__main__":
        main()
