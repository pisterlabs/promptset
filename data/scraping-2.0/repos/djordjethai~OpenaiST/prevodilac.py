# funkcije vezane za openai vision i gpt-4-turbo
# cita mp3, prikazije mp3, downloaduje mp3, cita sa slike, cita sa slike sa url, downloaduje opis slike
# cita iz txt, pdf i worda, cuva u txt, pdf i word i downloaduje plus mp3 preview i downloaduje mp3
# prevod sa raznih na razne jezike 

from openai import OpenAI
import streamlit as st
import os
import io
from myfunc.mojafunkcija import (
    st_style,
    positive_login,
    sacuvaj_dokument,
   )
from myfunc.asistenti import (
    audio_izlaz,
    priprema,
    )
import PyPDF2
from langchain.document_loaders import UnstructuredFileLoader
import re

st_style()
client = OpenAI()
version = "19.12.23."

# glavni program odredjuje ulazni dokument i jezike za prevodjenje
def main():
    st.markdown(
            f"<p style='font-size: 10px; color: grey;'>{version}</p>",
            unsafe_allow_html=True,
        )
    st.subheader("Prevodilac")  # Setting the title for Streamlit application
    with st.expander("Pročitajte uputstvo"):
         st.caption(
                """
                       ### Korisničko Uputstvo za Prevodioca


    1. **Ulazni dokument**
       - Po potrebi mozete konvertovati sliku ili audio zapis. Inace, mozete ucitati tekstualni dokument u .txt, .pdf ili .docx formatu.
       - Odaberite jezik ulaznog dokumenta.
       - Odaberije jezik izlaznog dokumenta
       - Pritisnite Submit
       - Učitajte fajl u odabranom formatu koji želite da prevedete. Za sliku i audio videćete i preview.
       - Unesite uputstvo za prevodjenje ili prihvatite default opciju.
       - Odaberite opciju "Glasovna naracija" po želji.
       
    2. **Čuvanje**
       - U levoj bočnoj traci možete sačuvati izlaz u txt, pdf ili docx formatu.
       - Tonski zapis možete sačuvati na samom playeru, desno, tri tačke, download.
       
    **Napomena:**
    - Za transkribovanje zvučnih zapisa koristi se OpenAI Whisper model. Zvučni zapis mora biti u .MP3 formatu i ne veći od 25Mb.
    - Za prevod teksta i citanje sa slika koristi se odgovarajući OpenAI GPT-4 model.
    

    Srećno sa korišćenjem alata za prevodjenje!  
                       """
            )
    

   
    if "final_content" not in st.session_state:
        st.session_state["final_content"] = "Prevod"
        
    st.subheader("Ulazni dokument")
    
    with st.sidebar:
        priprema()
    with st.form(key="ulaz", clear_on_submit=False):
 
        jezik_izlaza = st.selectbox("izaberite jezik izlaznog dokumenta", ("srpski", "english", "french", "german", "hungarian", "italian", "spanish"))    
        submit_button = st.form_submit_button(label="Submit")
         
    citaj_tekst(jezik_izlaza)
         
   
                          

# cuvaj dokument, prima tekst, ime fajla i cuva za download u txt, docx i pdf formatu


# cita tekst i upit a izlaz je mp3 player


# cita tekst i prevodi. Prima jezik izlaza i izlaz je prevedeni tekst
def citaj_tekst(jezik_izlaza):
    client=OpenAI()
    st.info("Čita tekst")
    uploaded_file = st.file_uploader(
        "Izaberite tekst za prevod",
        key="upload_file",
        type=["txt", "pdf", "docx"],
        help = "Odabir dokumenta",
    )

    if uploaded_file is not None:
        
        with io.open(uploaded_file.name, "wb") as file:
            file.write(uploaded_file.getbuffer())

        if ".pdf" in uploaded_file.name:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            num_pages = len(pdf_reader.pages)
            text_content = ""

            for page in range(num_pages):
                page_obj = pdf_reader.pages[page]
                text_content += page_obj.extract_text()
            text_content = text_content.replace("•", "")
            text_content = re.sub(r"(?<=\b\w) (?=\w\b)", "", text_content)
            with io.open("temp.txt", "w", encoding="utf-8") as f:
                f.write(text_content)

            loader = UnstructuredFileLoader("temp.txt", encoding="utf-8")
        else:
            # Creating a file loader object
            loader = UnstructuredFileLoader(file_path=uploaded_file.name, encoding="utf-8")

        result = loader.load()
                
      #  content = "Zapisnik"   
        with st.form(key="my_form", clear_on_submit=False):
            system_prompt=f"""You are a multi-lingual language expert. You must translate the text to the {jezik_izlaza} language and fix grammar \
and spelling errors but otherwise keep the text as is. 
"""
            opis = st.text_area(
                "Unesite instrukcije za sumarizaciju : ",
                system_prompt,
                key="prompt_prva",
                height=150,
                help = "Unos prompta."
            )
            audio_i = st.checkbox("Glasovna naracija")
            opis = f"{opis} {result[0].page_content}" 
            submit_button = st.form_submit_button(label="Submit")
            
            if submit_button:
                with st.spinner("Sačekajte trenutak..."):
                    response = client.chat.completions.create(
                    model="gpt-4-1106-preview",
                    temperature=0,
                    messages=[
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": opis 
                        }
                    ]
                )

                content =  response.choices[0].message.content
                st.session_state["final_content"] = content
            if st.session_state.final_content != "Prevod":
                if audio_i == True:
                    st.write("Glasovna naracija")
                    audio_izlaz(st.session_state.final_content)    
                with st.expander("Sažetak", True):
                    st.write(st.session_state.final_content)  # Displaying the summary
        with st.sidebar:      
            # if st.session_state.final_content !="Prevod":
            #     st.session_state["final_content"] = content
                sacuvaj_dokument(st.session_state.final_content, uploaded_file.name)
                
                    

# Deployment on Stremalit Login functionality
deployment_environment = os.environ.get("DEPLOYMENT_ENVIRONMENT")

if deployment_environment == "Streamlit":
    name, authentication_status, username = positive_login(main, " ")
else:
    if __name__ == "__main__":
        main()