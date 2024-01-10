import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.express as px
import leafmap.foliumap as leafmap
import openai
import tiktoken
import os 
import json
#Config must be first line in script
st.set_page_config(layout="wide")

# Set OpenAI API key
openai.organization = os.environ["OPENAI_ORGANIZATION"]
openai.api_key = os.getenv("OPENAI_API_KEY") 

max_input_tokens=7500
max_tokens_output=500
encoding = "cl100k_base"

if 'language' not in st.session_state:
    st.session_state.language = "🇩🇪 Deutsch"

# load translation data json
with open("data/translate_app.json", "r") as f:
    translator = json.load(f)



# calculate number of tokens in a text string
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# run gpt
def run_gpt(prompt, max_tokens_output, timeout=10):
    completion = openai.ChatCompletion.create(
      model = 'gpt-4',
      messages = [
        {'role': 'user', 'content': prompt}
      ],
      max_tokens = max_tokens_output,
      n = 1,
      stop = None,
      temperature=0.5,
      timeout=timeout
    )
    return completion['choices'][0]['message']['content']

# create start prompt
def start_prompt_creator(message_type, cluster):
    if len(cluster) > 1:
        cluster = ", ".join(cluster)
    else:
        cluster = cluster[0]
    if message_type == "Telegram":
            if st.session_state.language == "🇬🇧 English":
                start_prompt = f"looking at these telegram messages about {cluster} give a summary regarding the needs of refugees! Response in English!"
            if st.session_state.language == "🇩🇪 Deutsch":
                start_prompt = f"looking at this telegram messages about {cluster} what are the up to 5 top needs of refugees? Response in German Language"
            return start_prompt, cluster
    if message_type == "Twitter":
            if st.session_state.language == "🇬🇧 English":
                start_prompt = f"looking at this twitter messages about {cluster} what are the up to 5 to issues? If possibile focus on refugees. Response in English Language"
            if st.session_state.language == "🇩🇪 Deutsch":
                start_prompt = f"looking at this twitter messages about {cluster} what are the up to 5 to issues? If possibile focus on refugees. Response in German Language"
            return start_prompt, cluster
    if message_type == "News":
            if st.session_state.language == "🇬🇧 English":
                start_prompt = f"looking at this news articles about {cluster} what are the up to 5 to issues? If possibile focus on refugees. Response in English Language"
            if st.session_state.language == "🇩🇪 Deutsch":
                start_prompt = f"looking at this news articles about {cluster} what are the up to 5 to issues? If possibile focus on refugees. Response in German Language"
            return start_prompt, cluster

# sample from df
def sample_df_gpt_analysis(df, start_prompt, max_input_tokens):
    current_input_tokens = num_tokens_from_string(start_prompt, encoding_name=encoding)
    text_list = []
    text_list.append(start_prompt)
    while max_input_tokens > current_input_tokens:
        df_sample = df.sample(n=1, replace=False)
        df = df.drop(df_sample.index)
        current_input_tokens += df_sample["tokens"].values[0]
        if current_input_tokens > max_input_tokens:
            break
        text_list.append(df_sample["text"].values[0])
    
    text = '\n'.join(text_list)
    return text

# write output to streamlit
def write_output(text, summary_select, cluster):
    st.header(translator[st.session_state.language]["Your Summary 😊"])
    st.write(text)

# load geopandas data
gdf = gpd.read_file("data/europe.geojson")

#function dummy space in sidebar
def dummy_function_space():
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.write("\n")

def dummy_function_space_small():
    st.write("\n")

#functions to load data
@st.cache()
def load_telegram_data():
    df = pd.read_csv("data/df_telegram.csv")
    #print(df.head(1))
    df['date'] = pd.to_datetime(df['date'], utc=True).dt.date
    df = df[pd.to_datetime(df['date']) <= pd.to_datetime('2023-02-24')]
    return df
@st.cache
def load_news_data():
    df = pd.read_csv("data/df_news.csv")
    df['date'] = pd.to_datetime(df['date'], utc=True).dt.date
    return df
@st.cache()
def load_twitter_data():
    df = pd.read_csv("data/df_twitter.csv")
    df['date'] = pd.to_datetime(df['date'], utc=True).dt.date
    return df



# manipulate data
def create_df_value_counts(df):
    messages_per_week_dict = dict(df.value_counts("date"))
    df_value_counts = df.value_counts(["cluster", "date"]).reset_index()
    df_value_counts.columns = ["cluster", "date", "occurence_count"]
    return df_value_counts

def modify_df_for_table(df_mod, country_select, state_select, cluster_select, date_slider, metric_select=None):
    if country_select!=translator[st.session_state.language]["all countries analysed"]:
        df_mod = df_mod[df_mod.country==translator[st.session_state.language][country_select]]
    if state_select not in [translator[st.session_state.language]["all states analysed"]]:
        df_mod = df_mod[df_mod.state==state_select]
    if not translator[st.session_state.language]["all found topics"] in cluster_select:
        df_mod = df_mod[df_mod.cluster.isin(cluster_select)]
    df_mod = df_mod[df_mod.date.between(date_slider[0], date_slider[1])]
    return df_mod



# load data
df_telegram = load_telegram_data()
df_twitter = load_twitter_data()
df_news = load_news_data()
state_select = "all states analysed"
with st.sidebar:
    language_select = st.selectbox(
        'Language',
        options=["🇩🇪 Deutsch", "🇬🇧 English"],
        index=["🇩🇪 Deutsch", "🇬🇧 English"].index(st.session_state.language)
    )
    if st.session_state.language != language_select:
        st.session_state.language = language_select
    
    cluster_select_telegram = st.multiselect(
        translator[st.session_state.language]['Choose the topics of interest within the telegram data'],
        [translator[st.session_state.language]["all found topics"]] + df_telegram.cluster.unique().tolist(),
        [translator[st.session_state.language]["all found topics"]]
        )
    # cluster_select_twitter = st.multiselect(
    #     translator[st.session_state.language]['Choose the topics of interest within the twitter data'],
    #     [translator[st.session_state.language]["all found topics"]] + df_twitter.cluster.unique().tolist(),
    #     [translator[st.session_state.language]["all found topics"]]
    #     )
    # cluster_select_news = st.multiselect(
    #     translator[st.session_state.language]['Choose the topic of interest within the news data'],
    #     [translator[st.session_state.language]["all found topics"]] + df_news.cluster.unique().tolist(),
    #     [translator[st.session_state.language]["all found topics"]]
    #     )
    dummy_function_space()
    summary_select = st.selectbox(
        translator[st.session_state.language]['show summary of'],
        ["Telegram", "Twitter", "News"],
        )
    calculate_summary = st.button(translator[st.session_state.language]["prepare summary"])
    dummy_function_space_small()
    show_table = st.button(translator[st.session_state.language]['show data in table'])

st.title(translator[st.session_state.language]['Identification of the most relevant topics in the context of the Ukrainian Refugee Crisis in the media and social media'])

# create text columns for country, state and time selection
text_col1, text_col2  = st.columns(2)
with text_col1:
    country_select = st.selectbox(
        translator[st.session_state.language]["Select a country of interest"],
        [translator[st.session_state.language]["all countries analysed"], translator[st.session_state.language]["Germany"], translator[st.session_state.language]["Switzerland"]],
        )
with text_col2:
    date_slider = st.slider(translator[st.session_state.language]['Choose date range of interest'],
        min_value=df_telegram.date.min(), 
        value=(df_telegram.date.min(), df_telegram.date.max()), 
        max_value=df_telegram.date.max()
        )


df_telegram_mod = modify_df_for_table(df_mod=df_telegram, country_select=country_select, state_select=state_select, cluster_select=cluster_select_telegram, date_slider=date_slider)
df_value_counts_telegram = create_df_value_counts(df=df_telegram_mod)
# df_twitter_mod = modify_df_for_table(df_mod=df_twitter, country_select=country_select, state_select=state_select, cluster_select=cluster_select_twitter, date_slider=date_slider)
# df_value_counts_twitter = create_df_value_counts(df=df_twitter_mod)    
# df_news_mod = modify_df_for_table(df_mod=df_news, country_select=country_select, state_select=state_select, cluster_select=cluster_select_news, date_slider=date_slider)
# df_value_counts_news = create_df_value_counts(df=df_news_mod) 


visual_col1, visual_col2= st.columns(2)
with visual_col1:
    if country_select==translator[st.session_state.language]["all countries analysed"]:
        m = leafmap.Map(center=[46.449212, 7.734375], zoom=7,  height="45px", width="600px")
        m.add_gdf(gdf[gdf["country"].isin(["Switzerland", "Germany"])], layer_name="Countries choosen", fill_colors=["red"])
        m.to_streamlit()
    
    if country_select==translator[st.session_state.language]["Switzerland"] and state_select==translator[st.session_state.language]["all states analysed"]:
        m = leafmap.Map(center=[46.449212, 7.734375], zoom=7,  height="45px", width="600px")
        m.add_gdf(gdf[gdf["country"]!="Switzerland"], layer_name="Countries", fill_colors=["blue"])
        m.add_gdf(gdf[gdf["country"]=="Switzerland"], layer_name="Countries Choosen", fill_colors=["red"])
        m.to_streamlit()
    
    if country_select==translator[st.session_state.language]["Switzerland"] and state_select!=translator[st.session_state.language]["all states analysed"]:
        m = leafmap.Map(center=[46.449212, 7.734375], zoom=7,  height="45px", width="600px")
        m.add_gdf(gdf[gdf["state"]!=state_select], layer_name="Countries", fill_colors=["blue"])
        m.add_gdf(gdf[gdf["state"]==state_select], layer_name="Countries Choosen", fill_colors=["red"])
        m.to_streamlit()

    if country_select==translator[st.session_state.language]["Germany"] and state_select==translator[st.session_state.language]["all states analysed"]:
        m = leafmap.Map(center=[46.449212, 7.734375], zoom=7,  height="45px", width="600px")
        m.add_gdf(gdf[gdf["country"]=="Germany"], layer_name="Countries Choosen", fill_colors=["red"])
        m.add_gdf(gdf[gdf["country"]!="Germany"], layer_name="Countries", fill_colors=["blue"])
        m.to_streamlit()
    
    if country_select==translator[st.session_state.language]["Germany"] and state_select!=translator[st.session_state.language]["all states analysed"]:
        m = leafmap.Map(center=[46.449212, 7.734375], zoom=7,  height="45px", width="600px")
        m.add_gdf(gdf[gdf["state"]==state_select], layer_name="Countries Choosen", fill_colors=["red"])
        m.add_gdf(gdf[gdf["state"]!=state_select], layer_name="Countries", fill_colors=["blue"])
        m.to_streamlit()

    # if country_select==translator[st.session_state.language]["Germany"]s or country_select==translator[st.session_state.language]["Switzerland"] or country_select==translator[st.session_state.language]["all countries analysed"]:
    #     title_diagram_news = translator[st.session_state.language]["Topics over time on News within"] + " " + country_select
    #     fig = px.line(df_value_counts_news.sort_values(['date']), x="date", y="occurence_count", color='cluster', title=title_diagram_news)
    # else:
    #     title_diagram_news = translator[st.session_state.language]["Topics over time on News within"] + " " + state_select
    #     fig = px.line(df_value_counts_news.sort_values(['date']), x="date", y="occurence_count", color='cluster', title=title_diagram_news)
    # fig.update_xaxes(title_text=translator[st.session_state.language]["Date"])
    # fig.update_yaxes(title_text=translator[st.session_state.language]["Count"])
    # st.plotly_chart(fig, use_container_width=True)

with visual_col2:
    if country_select==translator[st.session_state.language]["Germany"] or country_select==translator[st.session_state.language]["Switzerland"] or country_select==translator[st.session_state.language]["all countries analysed"]:
        title_diagram_telegram = translator[st.session_state.language]["Topics over time on Telegram within"] + " " + country_select
        fig = px.line(df_value_counts_telegram.sort_values(['date']), x="date", y="occurence_count", color='cluster', title=title_diagram_telegram)
    else:
        title_diagram_telegram = translator[st.session_state.language]["Topics over time on News within"] + " " + state_select
        fig = px.line(df_value_counts_telegram.sort_values(['date']), x="date", y="occurence_count", color='cluster', title=title_diagram_telegram)
    fig.update_xaxes(title_text=translator[st.session_state.language]["Date"])
    fig.update_yaxes(title_text=translator[st.session_state.language]["Count"])
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("<p style='margin-top: 150px;'</p>", unsafe_allow_html=True)

    # if country_select==translator[st.session_state.language]["Germany"] or country_select==translator[st.session_state.language]["Switzerland"] or country_select==translator[st.session_state.language]["all countries analysed"]:
    #     title_diagram_twitter = translator[st.session_state.language]["Topics over time on Twitter within"] + " " + country_select
    #     fig = px.line(df_value_counts_twitter.sort_values(['date']), x="date", y="occurence_count", color='cluster', title=title_diagram_twitter)
    # else:
    #     title_diagram_twitter = translator[st.session_state.language]["Topics over time on Twitter within"] + " " + state_select
    #     fig = px.line(df_value_counts_twitter.sort_values(['date']), x="date", y="occurence_count", color='cluster', title=title_diagram_twitter)
    # fig.update_xaxes(title_text=translator[st.session_state.language]["Date"])
    # fig.update_yaxes(title_text=translator[st.session_state.language]["Count"])
    # st.plotly_chart(fig, use_container_width=True)

if calculate_summary:
    if summary_select=="Telegram":
        df_mod = df_telegram_mod
        cluster = cluster_select_telegram
    if summary_select=="Twitter":
        df_mod = df_twitter_mod
        cluster = cluster_select_twitter
    if summary_select=="News":
        df_mod = df_news_mod
        cluster = cluster_select_news

    dummy_text_summary = st.header(translator[st.session_state.language]["Creating your summary ⏳😊"])
    start_prompt, cluster_str = start_prompt_creator(message_type=summary_select, cluster=cluster)
    prompt = sample_df_gpt_analysis(df=df_mod, start_prompt=start_prompt, max_input_tokens=max_input_tokens-max_tokens_output)
    try:
        text = run_gpt(prompt, max_tokens_output, timeout=10)
    except openai.OpenAIError as e:
        text = translator[st.session_state.language]["Sorry, request timed out. Please try again."]
    dummy_text_summary.empty()
    write_output(text, summary_select, cluster_str)

if show_table:
    if summary_select=="Telegram":
        st.dataframe(df_telegram_mod) 
    if summary_select=="Twitter":
        st.dataframe(df_twitter_mod)
    if summary_select=="News":
        st.dataframe(df_news_mod)
