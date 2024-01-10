from annotated_text import annotated_text
from dotenv import load_dotenv
from matplotlib.figure import Figure
import openai
import pandas as pd
import streamlit as st
from streamlit_extras.switch_page_button import switch_page

# # Add the parent directory to the path
# BASE = Path(__file__).resolve().parent.parent
# if str(BASE) not in sys.path:
#     sys.path.append(str(BASE))

from scrape.absa.absa import create_absa_heatmap as _create_absa_heatmap

from scrape.absa.absa import get_annotated_absa
from scrape.types import OpenAIModel, GetDataOutput
from scrape.utils import get_data as _get_data
from scrape.analysis import create_wordcloud

# load_dotenv()

if "url" not in st.session_state or not st.session_state["url"]:
    switch_page("start")

if "api_key" not in st.session_state or not st.session_state["api_key"]:
    switch_page("start")

# CLIENT = openai.Client(api_key=os.environ["OPENAI_API_KEY"])
CLIENT = openai.Client(api_key=st.session_state["api_key"])
MAX_LENGTH = 2000  # The maximum number of tokens for the prompt


@st.cache_data
def get_data(url: str) -> GetDataOutput:
    return _get_data(CLIENT, url, max_length=MAX_LENGTH)


@st.cache_data
def create_absa_heatmap(df: pd.DataFrame) -> Figure:
    return _create_absa_heatmap(df)


# @st.cache_data
# def create_wordcloud(freq: Counter, width: int = 1280, height: int = 720) -> None:
#     _create_wordcloud(freq, width, height)


st.set_page_config(
    page_title="Sentiment Explorer",
    page_icon="🎭",
    layout="wide",
)


# reviews = get_movie_reviews(st.session_state['url'])
# url = "/home/richard/lope/dspy/lab14/examples/letterboxd.html"
# metadata, reviews = get_movie_reviews(url)


#######################
# 準備資料
#######################

# 載入資料
# if "url" not in st.session_state:
#     switch_page("start")
# if not st.session_state["url"]:
#     switch_page("start")


url = st.session_state["url"]
if not url:
    switch_page("start")
data = get_data(url)

# # heatmap
# heatmap = create_absa_heatmap(data.absa_counts_df)

#######################
# Streamlit 介面
#######################

# 設定分頁

with st.sidebar:
    reset = st.button("Reset", use_container_width=True, type="primary")
    if reset:
        st.session_state["url"] = ""
        switch_page("start")
summary_tab, table_tab, annotated_tab = st.tabs(
    ["Summary", "Details", "Annotated ABSA"]
)

with summary_tab:
    st.markdown(f"# {data.title}")
    st.markdown(data.summary)

with table_tab:
    st.markdown(f"# {data.title} Texts")

    st.markdown("## Word Cloud")
    create_wordcloud(data.word_counts)
    st.pyplot()

    st.markdown("## ABSA Heatmap")
    st.pyplot(
        create_absa_heatmap(data.absa_counts_df), dpi=1000, use_container_width=True
    )

    st.markdown("## Texts")

    df = data.df

    if data.source == "movie":
        assert data.df_filter
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            hide_spoilers = st.toggle("Hide Spoilers", value=False)
        with col2:
            hide_positive = st.toggle("Hide Positive", value=False)
        with col3:
            hide_negative = st.toggle("Hide Negative", value=False)
        with col4:
            hide_neutral = st.toggle("Hide Neutral", value=False)

        df = data.df_filter(
            hide_spoilers, hide_negative, hide_positive, hide_neutral, df
        )

    st.dataframe(df, use_container_width=True, height=1000)

with annotated_tab:
    st.markdown("# Annotate a Text")
    with st.form("annotate_form"):
        textbox = st.text_area("Review", height=200)
        submitted = st.form_submit_button("Submit")
        if submitted:
            annotated_text(
                get_annotated_absa(
                    client=CLIENT,
                    text=textbox,
                    aspects=data.aspect_list,
                    max_length=MAX_LENGTH,
                    model_name=OpenAIModel.GPT4,
                )
            )
