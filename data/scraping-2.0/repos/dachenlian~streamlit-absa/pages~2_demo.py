from dataclasses import asdict
import json
from pathlib import Path
import sys

from annotated_text import annotated_text
from dotenv import load_dotenv
import openai
import pandas as pd
import streamlit as st
from streamlit_extras.switch_page_button import switch_page

from scrape.types import LetterboxdReview, MovieMetadata, OpenAIModel

# Add the parent directory to the path
BASE = Path(__file__).resolve().parent.parent
if str(BASE) not in sys.path:
    sys.path.append(str(BASE))

from scrape.absa.absa import (
    count_absa,
    create_absa_counts_df,
    create_absa_df,
    create_absa_heatmap,
    get_absa,
    get_val_from_absa_output_key,
    get_annotated_absa,
)
from scrape.absa.aspects import MOVIE_ASPECTS
from scrape.absa.types import GetABSAOutput
from scrape.letterboxd import get_letterboxd_reviews
from scrape.imdb import get_imdb_reviews
from scrape.types import IMDbReview
from scrape.analysis import (
    create_wordcloud,
    create_word_count,
)

# load_dotenv()

if "url" not in st.session_state or not st.session_state["url"]:
    switch_page("start")

if "api_key" not in st.session_state or not st.session_state["api_key"]:
    switch_page("start")

# CLIENT = openai.Client(api_key=os.environ["OPENAI_API_KEY"])
CLIENT = openai.Client(api_key=str(st.session_state["api_key"]))

st.set_page_config(
    page_title="Movie Review Explorer",
    page_icon="🎬",
    layout="wide",
)


@st.cache_data
def get_demo_reviews() -> list[LetterboxdReview]:
    with BASE.joinpath(
        "examples/the-boy-and-the-heron-letterboxd-240-reviews.json"
    ).open() as f:
        return [LetterboxdReview(**d) for d in json.load(f)]


@st.cache_data
def get_demo_absa() -> GetABSAOutput:
    with BASE.joinpath(
        "examples/the-boy-and-the-heron-letterboxd-240-absa.json"
    ).open() as f:
        return json.load(f)


@st.cache_data
def get_demo_summary() -> str:
    with BASE.joinpath(
        "examples/the-boy-and-the-heron-letterboxd-240-summary.txt"
    ).open() as f:
        return f.read()


@st.cache_data
def get_demo_wordcloud():
    reviews = get_demo_reviews()
    word_counts = create_word_count([r.review for r in reviews])
    create_wordcloud(word_counts)
    st.pyplot()


@st.cache_data
def get_demo_heatmap():
    absa = get_demo_absa()
    absa_df = create_absa_counts_df(count_absa(absa), proportional=True)
    heatmap = create_absa_heatmap(absa_df)
    return heatmap


def get_movie_reviews(
    url: str,
) -> tuple[MovieMetadata, list[LetterboxdReview] | list[IMDbReview]]:
    if "letterboxd" in url:
        return get_letterboxd_reviews(url)
    elif "imdb" in url:
        return get_imdb_reviews(url)
    else:
        raise ValueError(f"Unknown website: {url}")


def filter_df(
    hide_spoilers: bool,
    hide_negative: bool,
    hide_positive: bool,
    hide_neutral: bool,
    df: pd.DataFrame,
) -> pd.DataFrame:
    if hide_spoilers:
        df = df[df["contains_spoilers"] == "No"]

    if hide_negative:
        df = df[~(df["negative"] != "")]

    if hide_positive:
        df = df[~(df["positive"] != "")]

    if hide_neutral:
        df = df[~(df["neutral"] != "")]

    return df


# reviews = get_movie_reviews(st.session_state['url'])
# url = "/home/richard/lope/dspy/lab14/examples/letterboxd.html"
# metadata, reviews = get_movie_reviews(url)


#######################
# 準備資料
#######################


# 載入資料
movie_title = "The Boy and the Heron"
reviews = get_demo_reviews()
absa = get_demo_absa()
summary = get_demo_summary()

word_counts = create_word_count([r.review for r in reviews])
absa_df = create_absa_df(absa)
contains_spoilers = get_val_from_absa_output_key(absa, "contains_spoilers")

# heatmap
# absa_df = create_absa_counts_df(count_absa(absa), proportional=True)
# heatmap = create_absa_heatmap(absa_df)

# 準備 dataframe
df = pd.DataFrame([asdict(r) for r in reviews])

# 建立新的 contains_spoilers 列
df["contains_spoilers"] = contains_spoilers
df["contains_spoilers"] = (
    df["contains_spoilers"].fillna(True).map({True: "Yes", False: "No"})
)

# 建立新的 rating 列
df["rating"] = df["rating"].fillna(-1)

# 合併 df 和 absa_df （positive, negative, neutral） 列
df = pd.merge(df, absa_df, left_index=True, right_index=True, how="outer")


#######################
# Streamlit 介面
#######################

with st.sidebar:
    reset = st.button("Reset", use_container_width=True, type="primary")
    if reset:
        st.session_state["url"] = ""
        switch_page("start")

# 設定分頁
summary_tab, table_tab, annotated_tab = st.tabs(
    ["Summary", "Details", "Annotated ABSA"]
)


with summary_tab:
    st.markdown("# The Boy and the Heron")
    st.markdown(summary)

with table_tab:
    st.markdown("# The Boy and the Heron Reviews")

    st.markdown("## Word Cloud")
    get_demo_wordcloud()
    # create_wordcloud(word_counts)
    # st.pyplot()

    st.markdown("## ABSA Heatmap")
    st.pyplot(get_demo_heatmap(), dpi=1000, use_container_width=True)

    st.markdown("## Reviews")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        hide_spoilers = st.toggle("Hide Spoilers", value=False)
    with col2:
        hide_positive = st.toggle("Hide Positive", value=False)
    with col3:
        hide_negative = st.toggle("Hide Negative", value=False)
    with col4:
        hide_neutral = st.toggle("Hide Neutral", value=False)

    df = filter_df(hide_spoilers, hide_negative, hide_positive, hide_neutral, df)

    st.dataframe(df, use_container_width=True, height=1000)

with annotated_tab:
    st.markdown("# Annotate a Review")
    with st.form("annotate_form"):
        textbox = st.text_area("Review", height=200)
        submitted = st.form_submit_button("Submit")
        if submitted:
            annotated_text(
                get_annotated_absa(
                    client=CLIENT,
                    text=textbox,
                    aspects=MOVIE_ASPECTS,
                    model_name=OpenAIModel.GPT4,
                )
            )
