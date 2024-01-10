import streamlit as st
st.set_page_config(
    page_title="k_tranditional_drink",
    layout="wide",
    initial_sidebar_state="collapsed"
)
st.markdown(
    """
<style>
    [data-testid="collapsedControl"] {
        display: none
    }
</style>
""",
    unsafe_allow_html=True,
)
st.markdown(
    """
    <style>
    .css-1jc7ptx, .e1ewe7hr3, .viewerBadge_container__1QSob,
    .styles_viewerBadge__1yB5_, .viewerBadge_link__1S137,
    .viewerBadge_text__1JaDK {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)
from streamlit import components
import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import openai
openai.api_key = st.secrets.OPENAI_TOKEN
from supabase import create_client
import pickle
from streamlit_extras.switch_page_button import switch_page


@st.cache_resource(show_spinner=None)
def init_connection():
    url = st.secrets["SUPABASE_URL"]
    key = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase_client = init_connection()

EMBEDDING_MODEL = "text-embedding-ada-002"

st.subheader(":house:", anchor="main")
empty0, con0, empty7 = st.columns([0.3, 1.0, 0.3])
with empty0:
    st.empty()

with con0:
    st.image("./f_image/title_03.png")

with empty7:
    st.empty()

@st.cache_resource(show_spinner=None)
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

container = st.container()
with container:
    empty1, con1, empty2 = st.columns([0.3, 1.0, 0.3])
    with empty1:
        st.empty()

    with con1:
        st.image("./f_image/main_text.png")
        e1, e2, e3 = st.columns(3)
        with e2:
            local_css("./button_style.css")
            want_to_contribute = st.button("나만의 전통주 찾기")
            if want_to_contribute:
                switch_page("k_trenditonal_drinks")

    with empty2:
        st.empty()

#STEP 2) 데이터 로드
@st.cache_resource(show_spinner=None)
def load_data():
    feature_df = pd.read_csv("./data/feature_total_f.csv", encoding="utf-8")
    main_df = pd.read_csv("./data/main_total_no_features_f.csv", encoding="utf-8")
    ingredient_df = pd.read_csv("./data/ingredient_total_id_f.csv", encoding="utf-8")
    embedding_df = pd.read_csv("./data/embedding_f.csv", encoding="utf-8")
    emoji_df = pd.read_csv("./data/emoji_selected_f.csv", encoding="utf-8")
    food_df = pd.read_csv("./data/food_preprocessed_f.csv", encoding="utf-8-sig")
    return feature_df, main_df, ingredient_df, embedding_df, emoji_df, food_df

feature_df, main_df, ingredient_df, embedding_df, emoji_df, food_df = load_data()

@st.cache_resource(show_spinner=None)
def embedding_c():
    embeddings = [np.array(eval(embedding)).astype(float) for embedding in embedding_df["embeddings"].values]
    stacked_embeddings = np.vstack(embeddings)

    return stacked_embeddings

stacked_embeddings = embedding_c()

#STEP 3) 캐시 불러오고 임베딩 저장하기
embedding_cache_path = "./data/recommendations_embeddings_cache.pkl"

try:
    embedding_cache = pd.read_pickle(embedding_cache_path)
except FileNotFoundError:
    embedding_cache = {}
with open(embedding_cache_path, "wb") as embedding_cache_file:
    pickle.dump(embedding_cache, embedding_cache_file)

