# import libraries
import streamlit as st
from annotated_text import annotated_text
import re
import pandas as pd
from matplotlib.backends.backend_agg import RendererAgg
import spacy
import openai
import os
from dotenv import load_dotenv
from nltk.data import find
import gensim
import pickle 

# get openai api key
load_dotenv()
OPENAI_APIKEY = os.getenv('OPENAI_APIKEY')

# Set page title and favicon.
st.set_page_config(
    page_title="Vacancy writer", page_icon="writing_hand", layout="wide",
)

_lock = RendererAgg.lock

# import NER model Spacy
@st.cache(show_spinner=False, allow_output_mutation=True, suppress_st_warning=True)
def load_models():
    dutch_model = spacy.load("./models/NL/nl_core_news_sm/nl_core_news_sm-3.2.0/")
    english_model = spacy.load("./models/ENG/en_core_web_sm/en_core_web_sm-3.2.0/")
    models = {"ENG": english_model, "NL": dutch_model}
    return models

# import suggestion model NLTK
word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))
model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)

# Define question answering model TODO remove apikey
@st.cache(show_spinner=False, allow_output_mutation=True, suppress_st_warning=True)
def gpt3(stext):
    openai.api_key = OPENAI_APIKEY
    response = openai.Completion.create( 
        engine="davinci-instruct-beta", 
        prompt=stext, 
                temperature=0.1, 
                max_tokens=1000,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0
    )
    content = response.choices[0].text.split('.') 
    return response.choices[0].text 

# process text input for regex patterns
def process(user_input):
    list = []
    for w in re.split(seperators, user_input.lower()):
        if any(regex.match(w) for regex in pattern_list):
            list.append((w, "non-inclusive", "#faa"))
        else:
            list.append(" " + w + " ")
    return list

# processing text input for NER
def process_text(doc, selected_entities, anonymize=False):
    tokens = []
    for token in doc:
        if (token.ent_type_ == "PERSON") & ("PER" in selected_entities):
            tokens.append((token.text, "Person", "#faa"))
        elif (token.ent_type_ in ["GPE", "LOC"]) & ("LOC" in selected_entities):
            tokens.append((token.text, "Location", "#fda"))
        elif (token.ent_type_ == "ORG") & ("ORG" in selected_entities):
            tokens.append((token.text, "Organization", "#afa"))
        else:
            tokens.append(" " + token.text + " ")
    return tokens

#process for classifying vacancies
def vac_classification(user_input):
    # load the model from disk
    final_model = 'imb_model.pkl'
    loaded_model = pickle.load(open(final_model, 'rb'))
    #prediction for the vacancy
    pred  = loaded_model.predict_proba([user_input])
    if pred[0][0] > pred[0][1]:
        result = "**Non Discriminative**"
    else:
        result = "**Discriminative**"
    perc = str(round(pred[0][1]*100,2))+"%"
    return result, perc


# ---- layout -----
# sidebar
st.sidebar.title("J4 Vacancy Analysis App")

st.sidebar.image("https://images.unsplash.com/photo-1533299150403-a196e9ae00ea?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1508&q=80", 
    width=300,
)

st.sidebar.info("⚙️ Here you can adjust the setting for analyzing your vacancy text.")

st.sidebar.write("## Settings")
selected_language = st.sidebar.selectbox("Select a language", options=["ENG", "NL"])
selected_entities = st.sidebar.multiselect(
    "Select the entities you want to detect",
    options=["LOC", "PER", "ORG"],
    default=["LOC", "PER", "ORG"],
)

row0_spacer1, row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns(
    (.1, 2, .2, 1, .1))

st.title('Analyze your vacancy')

with row0_2:
    st.write('')

row0_2.subheader('')

row1_spacer1, row1_1, row1_spacer2 = st.columns((.1, 3.2, .1))

with row1_1:
    st.markdown("Hey there! Welcome to J4 Vacancy Analysis App. This app will detect non-inclusive words in vacancies and makes suggestions that will improve your vacancy")
    st.markdown(
        "**To begin, please enter the vacancy your are writing.** 👇")

user_input = st.text_area(
        "Input your own vacancy")

uploaded_file = st.file_uploader("or Upload a file", type=["doc", "docx", "pdf", "txt"])
if uploaded_file is not None:
    text_input = uploaded_file.getvalue()
    text_input = text_input.decode("utf-8")

# Data for highlight non-inclusive words
seperators = "\? |\."

flexible_orange = open("./data/regex_list.txt", "r")

pattern_list = []
for line in flexible_orange:
  line = re.sub('\n', '', line)
  line = re.compile(line)
  pattern_list.append(line)

flexible_orange.close()

row3_space1, row3_1, row3_space2, row3_2, row3_space3 = st.columns(
    (.1, 1, .1, 1, .1))

with row3_1, _lock:
    res,perc = vac_classification(user_input)
    st.subheader('Vacancy non-inclusiveness percentage')
    st.write("Your vacancy is classified as:  " +res)
    st.metric("Non-inclusiveness percentage:  ",value=perc)

with row3_2, _lock:
    st.subheader('Non-inclusive words')
    tokens = process(user_input)
    annotated_text(*tokens)

st.text("")
st.text("")

row4_space1, row4_1, row4_space2, row4_2, row4_space3 = st.columns(
    (.1, 1, .1, 1, .1))

with row4_1, _lock:
    st.subheader('Suggestions')
    st.write("If you want to get a suggestion that is more non-inclusive, just copy the highlighted part and paste it into here.")
    suggestion_input = st.text_input("Suggestion input")

with row4_2, _lock:
    st.text("")
    st.text("")
    st.text("")
    st.text("")
    st.markdown("**Your suggestion for '"+suggestion_input+"' will be:**")
    # making suggestion gpt3 model
    st.markdown("_Based on GPT-3:_")
    initial_query = "What is another word for "
    response = gpt3(initial_query+suggestion_input+"?")
    st.write(response)
    # making suggestion NLTK model
    st.markdown("_Based on NLTK model:_")
    st.table(pd.DataFrame(model.most_similar(positive=[suggestion_input], topn = 5), columns=['Suggestion','Percentage similar']))
    
st.markdown("""---""")

st.subheader('Named Entity Recognition')

models = load_models()

selected_model = models[selected_language]

doc = selected_model(user_input)
tokens = process_text(doc, selected_entities)

annotated_text(*tokens)