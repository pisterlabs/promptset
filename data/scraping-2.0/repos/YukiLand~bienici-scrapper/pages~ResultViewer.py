import streamlit as st
import pandas as pd
from openaiDescription import text_generator

st.set_page_config(
    page_title="House Finder Consulting",
    page_icon='🔎',
    layout="wide"
)

st.title("House Finder Consulting")  



    # prompt = "Fais moi un résumé de la ville de " + st.session_state["pattern"]


if 'articles' in st.session_state and  st.session_state["articles"] != []:
    prompt = "Fais moi un résumé en moins de 150 token de la ville de "  + st.session_state["pattern"]

    st.write("### Description de la ville")
    st.write(text_generator(prompt))

    df = pd.DataFrame(st.session_state["articles"])
    df.to_csv('articles.csv')
    st.download_button(
        label="Télécharger les annonces",
        data=df.to_csv().encode('utf-8'),
        file_name=st.session_state["pattern"] + '.csv',
        mime='text/csv'
    )

    for article in st.session_state["articles"]:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image(article['img'], width=200)
        with col2:
            st.write("###", article['name'])
            st.write("####", article['address'])
            st.write("####", article['price'])
            st.write("####", article['sqprice'])
            st.write("####", article['link'])
        st.write("---")

else:
    st.write("### Aucun résultat trouvé")