import streamlit as st
from implementation import MediaBiasModel
import pandas as pd
import plotly.express as px

import openai 


@st.cache_resource
def run_model(article):
    if len(article) > 5000:
        raise ValueError("The pasted text is too long. Consider shortening it or analyzing it in chunks!")

    elif len(article) < 24:
        raise ValueError("The pasted text is too short. There's a solid chance for embeddings with OpenAI to fail.")
    
    else:
        with st.spinner("Loading pre-trained classifiers..."):
            model = MediaBiasModel()
            model.load_models(model_tag='test', directory='models')

        with st.spinner("Prediciting labels..."):
            sentences = article.replace('\n','').replace('U.S.','United States').split('.')
            df_sentences = pd.DataFrame(sentences, columns=['text'])
            df_sentences['text_length'] = df_sentences['text'].apply(lambda x: len(x))
            df_sentences = model.predict_labels_df(df_sentences[df_sentences['text_length']>1])

    return df_sentences

def tryityourself_tab():

    st.header("Content analyzer")
    st.write("As a practical application of my model, I built an article analyzer which can take a larger body of text, analyze each sentence independently and return a summary of the article's topic, objectivity and political bias.")
    st.write("If you have an OpenAI API key, you can paste it here to use the content analyzer!")
    api_key = st.text_input(label='OpenAI API key', value='', key='tryitapi')
    if len(api_key)>0:
        openai.api_key = api_key
        st.write("Keep in mind that the topic classifier, in particular, will mostly work well on topics present in the training data, like environment, coronavirus and vaccines, abortion, politics, etc. Content pertaining to a very specific topic outside of these is more likely to be misclassified.")
        st.write("See it in action by pasting text below (limited to about a page to avoid overcharging the OpenAI API key and run faster in real time!)")

        article = st.text_area(label='Paste your content here', value="")


        if st.button("Analyze content"):
            df_sentences = run_model(article)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('*Topic*')
                st.subheader(df_sentences['topic'].mode().values[0])
            with col2:
                st.markdown('*Bias*')
                st.subheader(df_sentences['label_bias'].mode().values[0])
            with col3:
                st.markdown('*Political bias*')
                st.subheader(df_sentences['outlet_bias'].mode().values[0])
            st.divider()
            hist_fig_11 = px.histogram(df_sentences, x='topic', title='Distribution of topics')
            hist_fig_21 = px.histogram(df_sentences, x='label_bias', title='Distribution of bias labels')
            hist_fig_31 = px.histogram(df_sentences, x='outlet_bias', title='Distribution of outlet bias labels')

            st.write("The charts below show the distribution of topics, bias and outlet bias labels identified in the pasted content.")
            st.plotly_chart(hist_fig_11)
            st.plotly_chart(hist_fig_21)
            st.plotly_chart(hist_fig_31)


            


