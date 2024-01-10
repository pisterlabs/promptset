import json
import openai
import streamlit as st
from myfunc.mojafunkcija import open_file, init_cond_llm
import io


st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] {
            width: 400px !important; # Set the width to your desired value
        }
    </style>
    """,
    unsafe_allow_html=True,
)

def pripremaft():
    with st.sidebar:
        model, temp = init_cond_llm()
    
    final_placeholder = st.empty()
    placeholder = st.empty()

    with placeholder.container():
        input_file_path = st.file_uploader(
            "Izaberite fajl sa pitanjima", key="upload_pitanja", type='txt', help="Pitanja koja ste sastavili u prethodnom koraku.")

        source_file_path = st.file_uploader(
            "Izaberite fajl sa izvorom (ako postoji)", key="upload_izvor", type='txt', help="Fajl koje želite da vam bude izvor informacija za odgovore.")

        if input_file_path is not None:
            # Loading text from the file
            with io.open(input_file_path.name, "wb") as file:
                file.write(input_file_path.getbuffer())
            pitanje = open_file(input_file_path.name)

        if source_file_path is not None:
            # Loading text from the file
            with io.open(source_file_path.name, "wb") as file:
                file.write(source_file_path.getbuffer())
            prompt_source = open_file(source_file_path.name)
        else:
            prompt_source = ""
        with st.form(key='my_form'):
            sys_message = st.text_area(
                "Unesite sistemsku poruku: ", help="Opišite ponašanje i stil modela. Uključite i ime")
            izvor = st.text_input("Unesite naziv FT modela:",
                                help="Dajte ime modelu koji kreirate")
            
            submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        skinuto = False
        file_name = "izv.jsonl"
        with st.spinner('Kreiram odgovore...'):

            # source_file_path za odgovore koji se baziraju na specificnom tekstu, obuhvata prefix kao "Based on this text",
            # zatim sam text i na kraju sufix kao "Answer this question in your writing style:"
            # ako nije potrebno ucitava se empty.txt

            # Read questions from the input file
           
            questions = pitanje.splitlines()
            total_questions = len(questions)
            # Generate answers for each question, model and temp can be adjusted
            # Iterate through questions with index
            qa_answers_dict = {}

            for idx, question in enumerate(questions, 1):
                current_question_number = idx  # Get the current question number

                prompt = prompt_source + question

                response = openai.ChatCompletion.create(
                    model=model,
                    temperature=temp,
                    messages=[
                        {"role": "system", "content": sys_message},
                        {"role": "user", "content": prompt}
                    ]
                )

                answer = response.choices[0].message["content"].strip().replace('\n', '')

                # Add the answer to the dictionary with a key indicating the question number
                qa_answers_dict[f"Question {current_question_number}"] = answer

                with placeholder.container():
                    st.subheader(
                        f"Obrađujem {current_question_number}. od ukupno {total_questions} pitanja")
                    st.info(f"Pitanje: {question}")
                    st.success(f"Odgovor: {answer}")
            
            placeholder.empty()

            with final_placeholder.container():
                messages_list = []

                for user_content, assistant_content in zip(questions, qa_answers_dict.values()):
                    system_message = {"role": "system", "content": sys_message}

                    user_msg = {"role": "user", "content": user_content}
                    assistant_msg = {"role": "assistant", "content": assistant_content}
                    messages = [system_message, user_msg, assistant_msg]

                    # Append the JSON representation of messages with a newline character
                    messages_list.append(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")

                # Create a single JSONL formatted string by joining all the messages
                jsonl_string = "".join(messages_list)

                # Now, messages_list contains the JSON representation of messages with newlines


                st.subheader(
                    f"Obrada je završena")
                
                skinuto = st.download_button(
                "Download JSON",
                data=jsonl_string,
                file_name=f"{izvor}.jsonl",
                mime="application/json",
            )
            if skinuto:
                st.success(f"Tekstovi sačuvani na {file_name} su sada spremni za Embeding")



