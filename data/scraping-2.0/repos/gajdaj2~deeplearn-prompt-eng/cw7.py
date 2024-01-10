import openai
import streamlit as st


def get_name(plec):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Jesteś moim asystentem testera"
            },
            {
                "role": "user",
                "content": "Wygeneruj tylko jedno imie dla " + plec + " oczekiwany format {imie:[imie]}"
            }
        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response.choices[0].message.content


def get_konto(bank):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Jesteś moim asystentem testera"
            },
            {
                "role": "user",
                "content": "Wygeneruj mi numer konta dla banku " + bank + " oczekiwany format {numer_konta:[numer_konta]}"
            }
        ],
        temperature=1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    print(response)
    return response.choices[0].message.content


openai.api_key = "sk-#"

st.title("Generator danych osobowych")

plec = st.radio("Wybierz płeć", ("Kobieta", "Mężczyzna"))
generuj_imie = st.button("Generuj imie")
numer_konta = st.selectbox("Wybierz bank", ("Wybierz ...", "PKO", "ING", "Millenium", "Pekao"))
generuj_konto = st.button("Generuj numer konta")

if generuj_imie:
    st.text(get_name(plec))

if generuj_konto:
    st.text(get_konto(numer_konta))
