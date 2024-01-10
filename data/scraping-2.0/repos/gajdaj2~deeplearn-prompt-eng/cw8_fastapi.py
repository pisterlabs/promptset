import openai
import streamlit as st

st.title("Generator API testów")
openai.api_key = "######"
language = st.selectbox("Wybierz rodzaj frameworka", ("RestAssured","RestSharp"))

json = st.text_area("Wprowadź zawartoś pliku JSON")

if st.button("Generuj testy"):
    with st.spinner("Generowanie testów ..."):
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Jesteś asystentem testera automatycznego pomożesz mi wygenerować testy API"
                },
                {
                    "role": "user",
                    "content": "Wygeneruj testy automatyczne w " + language + " dla serwisów z pliku Json: "+json
                }
            ],
            temperature=1,
            max_tokens=256,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

    if language == "RestAssured":
        st.code(response.choices[0].message.content, language='csharp')
    else:
        st.code(response.choices[0].message.content, language='java')

    st.success("Testy zostały wygenerowane")
