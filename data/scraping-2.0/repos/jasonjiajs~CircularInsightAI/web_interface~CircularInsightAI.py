# Run locally with streamlit run CircularInsightAI.py

import streamlit as st
from openai import OpenAI
import pandas as pd
import numpy as np
import os
import json
import preprocessing
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
import random
from modal_streamlit import Modal

with st.sidebar:
    st.write("## Enter your OpenAI API key")
    openai_api_key = st.text_input("OpenAI API Key", type="password")
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"

    st.write("## How to use this app")
    st.markdown("""
        - Use dark mode for better readability
        - Gain context from the current macro outlook
        - Upload a CSV file with your ideas  
        - Filter out irrelevant and unclear ideas  
        - Explore remaining ideas, ranked by overall quality  
        - Click on an idea for in-depth analysis  
        - Export best ideas to a CSV file  
        - Share your circular economy business ideas with us!""")

    "[View source code](https://github.com/jasonjiajs/CircularInsightAI)"

# Set up OpenAI client
if os.environ.get("OPENAI_API_KEY") is not None:
    openai_api_key=os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
else:
    if openai_api_key is not None:
        client = OpenAI(api_key=openai_api_key)

st.title("📝 CircularInsightAI")
st.caption("🚀💡♻️ An AI-powered circular economy idea evaluation assistant")

# Display information on current and projected market sizes
st.write("### Macro Outlook: Financial projections for circular economy industries")
chart_data = pd.DataFrame(
{
    "Category": ["Current (2023)"] * 4 + ["Projected (2030)"] * 4,
    "Market size ($B)": [73.5, 57.8, 69.4, 6.5, 106.7, 244.6, 120.0, 18.27],
    "Industry": ["Food waste", "E-waste", "Plastic waste", "Clothing waste", "Food waste", "E-waste", "Plastic waste", "Clothing waste"]
}
)

col1, col2 = st.columns(2)
with col1:
    st.write("Market sizes: Current (2023) and Projected (2030)")
    fig, ax = plt.subplots(figsize=(8,3))
    sns.barplot(x="Industry", y="Market size ($B)", hue="Category", data=chart_data)
    st.pyplot(fig) # Adjust the height as needed

# Pie Chart for Current Market Share
with col2:
    col3, col4 = st.columns(2)
    with col3:
        st.metric(label="Food waste (2023)", value='$73.5B', delta='CAGR: 5.4%',
        delta_color="normal")
        st.metric(label="Plastic waste (2023)", value='$69.4B', delta='CAGR: 8.1%',
        delta_color="normal")
    with col4:
        st.metric(label="E-waste (2023)", value='$57.8B', delta='CAGR: 15.7%',
        delta_color="normal")
        st.metric(label="Clothing waste (2023)", value='$6.5B', delta='CAGR: 10.9%',
        delta_color="normal")

st.write("Sources: \
[Food waste](https://www.grandviewresearch.com/industry-analysis/food-waste-management-market), \
[E-waste](https://www.alliedmarketresearch.com/e-waste-management-market), \
[Plastic waste](https://www.marketsandmarkets.com/Market-Reports/recycled-plastic-market-115486722.html), \
[Clothing waste](https://www.futuremarketinsights.com/reports/clothing-recycling-market)")

st.write("### Unleash your ideas")
uploaded_file = st.file_uploader("Upload the csv file", type=("csv"))

if uploaded_file and not openai_api_key:
    st.info("Please add your OpenAI API key to continue.")

# Functions
@st.cache_data(show_spinner=False)
def split_frame(input_df, rows):
    df = [input_df.loc[i : i + rows - 1, :] for i in range(0, len(input_df), rows)]
    return df

@st.cache_data(show_spinner=False)
def get_df_with_filter_metrics(uploaded_file):
    df_full, df = preprocessing.read_data(uploaded_file, nrows_to_keep=None) # reduce the number of rows during development, e.g. nrows_to_keep=5
    df_full = preprocessing.get_category(df_full)
    df_with_filter_metrics = preprocessing.get_metrics_for_filtering_ideas(df_full, df, client, finetuned=True)
    return df_full, df, df_with_filter_metrics

@st.cache_data(show_spinner=False)
def get_df_metrics(df_with_filter_metrics, filter_score):
    # Filter to keep on the ideas with a filter score above a certain threshold
    df_full = df_with_filter_metrics[df_with_filter_metrics['filter_score'] >= filter_score]
    df_full['details'] = False
    df = df_full[['problem', 'solution']]
    df_metrics = preprocessing.get_metrics_for_ranking_ideas(df_full, df, client, finetuned=False)
    return df_metrics

def get_filter_message(df_with_filter_metrics, score, score_type):
    score_var = score_type + '_score'
    df_filtered = df_with_filter_metrics.loc[df_with_filter_metrics[score_var] >= score]
    filtered_nrows, nrows, pct = df_filtered.shape[0], df_with_filter_metrics.shape[0], round(df_filtered.shape[0] / df_with_filter_metrics.shape[0] * 100, 1)
    message = f"With a {score_type} score of {score}, you will keep {filtered_nrows} out of {nrows} ideas, or {pct}% of all ideas."
    return message

# Part 1
if uploaded_file and openai_api_key:
    df_full, df, df_with_filter_metrics = get_df_with_filter_metrics(uploaded_file)
    # Show distribution of filter_score
    st.write("## 1. Filter out irrelevant and unclear ideas") 
    st.write("The filter score ranges from 1-3 and is a measure of how relevant and clear the idea is. The higher the score, the more relevant and clear the idea is. We recommend only keeping ideas with a filter score of 2.5 or higher.")    

    # Leave space for filter score message
    filter_message = st.container()

    # Filter score top menu
    filter_score_top_menu = st.columns([0.3, 0.7])
    with filter_score_top_menu[0]:
        filter_score = st.slider('Filter Score', min_value=2.0, max_value=3.0, value=2.5, step=0.1)
    with filter_score_top_menu[1]:
        bar_chart = alt.Chart(df_with_filter_metrics).mark_bar().encode(
                x=alt.X("filter_score:O", bin=alt.Bin(maxbins=10)),
                y="count():Q",
                color="category:N"
            )
        st.altair_chart(bar_chart, use_container_width=True)

    # Data with filter metrics
    # Top menu
    top_menu = st.columns(3)
    with top_menu[0]:
        category_filter = st.multiselect("Filter by Category", options=df_with_filter_metrics['category'].unique(), default=df_with_filter_metrics['category'].unique(), key='category_filter_1')
    with top_menu[1]:
        sort_field = st.selectbox("Sort By", options=df_with_filter_metrics.columns, index=df_with_filter_metrics.columns.get_loc('filter_score'), key='sort_field_1')
    with top_menu[2]:
        sort_direction = st.radio("Direction", options=["⬆️", "⬇️"], horizontal=True, index=1, key='sort_direction_1')
    df_with_filter_metrics_interface = df_with_filter_metrics[df_with_filter_metrics['category'].isin(category_filter)]
    df_with_filter_metrics_interface = df_with_filter_metrics_interface.sort_values(by=sort_field, ascending=sort_direction == "⬆️", ignore_index=True)

    # Print message
    message = get_filter_message(df_with_filter_metrics, filter_score, 'filter')
    filter_message.info(message)

    # Table
    pagination = st.container()

    # Bottom Menu
    bottom_menu = st.columns((4, 1, 1))
    with bottom_menu[2]:
        batch_size = st.selectbox("Page Size", options=[25, 50, 100], key='batch_size_1')
    with bottom_menu[1]:
        total_pages = (
            int(len(df_with_filter_metrics_interface) / batch_size) if int(len(df_with_filter_metrics_interface) / batch_size) > 0 else 1
        )
        current_page = st.number_input(
            "Page", min_value=1, max_value=total_pages, step=1, key='current_page_1'
        )
    with bottom_menu[0]:
        st.markdown(f"Page **{current_page}** of **{total_pages}** ")

    pages = split_frame(df_with_filter_metrics_interface, batch_size)
    if df_with_filter_metrics_interface.shape[0] > 0:
        pagination.write("View the data with the filter scores below:")
        pagination.dataframe(data=pages[current_page - 1], use_container_width=True, hide_index = True)
    else:
        pagination.error("Select at least one category.")

# Part 2
# Define session state and Modal for pop-up
st.session_state["editor_key"] = random.randint(0, 100000)
modal = Modal(key="Demo Key", title="Selected Idea", padding=20,  max_width=900)

if uploaded_file and openai_api_key:
    st.write("## 2. Explore remaining ideas, ranked by overall quality")

    # Get filter metrics for remaining rows
    df_metrics = get_df_metrics(df_with_filter_metrics, filter_score)

    # Leave space for overall score message
    overall_message = st.container()

    # Overall score top menu
    overall_score_top_menu = st.columns([0.3, 0.7])
    with overall_score_top_menu[0]:
        overall_score = st.slider('Filter by Overall Score', min_value=1.0, max_value=5.0, value=4.0, step=0.1)
    with overall_score_top_menu[1]:
        bar_chart = alt.Chart(df_metrics).mark_bar().encode(
                x=alt.X("overall_score:O", bin=alt.Bin(maxbins=10)),
                y="count():Q",
                color="category:N"
            )
        st.altair_chart(bar_chart, use_container_width=True)

    # Top menu
    df_metrics_interface_table_columns = ['id','problem','solution','category','overall_score', 'market_potential', 'feasibility', 'scalability', 'innovation', 'alignment', 'novelty',
    'filter_score', 'relevance_problem', 'clarity_problem','suitability_solution','clarity_solution']
    top_menu = st.columns(3)
    with top_menu[0]:
        category_filter = st.multiselect("Filter by Category", options=df_metrics['category'].unique(), default=df_metrics['category'].unique(), key='category_filter_2')
    with top_menu[1]:
        sort_field = st.selectbox("Sort By", options=df_metrics_interface_table_columns, index=df_metrics_interface_table_columns.index('overall_score'), key='sort_field_2')
    with top_menu[2]:
        sort_direction = st.radio("Direction", options=["⬆️", "⬇️"], horizontal=True, index=1, key='sort_direction_2')
    
    df_metrics_interface = df_metrics[df_metrics['overall_score'] >= overall_score]
    df_metrics_interface = df_metrics_interface[df_metrics['category'].isin(category_filter)]
    df_metrics_interface = df_metrics_interface.sort_values(by=sort_field, ascending=sort_direction == "⬆️", ignore_index=True)

    # Print message
    message = get_filter_message(df_metrics, overall_score, 'overall')
    overall_message.info(message)

    # Table
    pagination = st.container()

    # Bottom Menu
    bottom_menu = st.columns((4, 1, 1))
    with bottom_menu[2]:
        batch_size = st.selectbox("Page Size", options=[25, 50, 100], key='batch_size_2')
    with bottom_menu[1]:
        total_pages = (
            int(len(df_metrics_interface) / batch_size) if int(len(df_metrics_interface) / batch_size) > 0 else 1
        )
        current_page = st.number_input(
            "Page", min_value=1, max_value=total_pages, step=1, key='current_page_2'
        )
    with bottom_menu[0]:
        st.markdown(f"Page **{current_page}** of **{total_pages}** ")

    pages = split_frame(df_metrics_interface, batch_size)

    # Pop-up
    def get_row_and_clear_selection():
        key = st.session_state["editor_key"]
        df = st.session_state["data"]
        selected_rows = st.session_state[key]["edited_rows"]
        selected_rows = [int(row) for row in selected_rows if selected_rows[row]["details"]]
        with modal.container():
            data_query = df.iloc[selected_rows]
            columns = ['problem','solution','category', 'overall_score', 
                       'market_potential', 'market_potential_eval', 'market_potential_advice',
                         'feasibility', 'feasibility_eval', 'feasibility_advice',
                         'scalability', 'scalability_eval', 'scalability_advice',
                         'innovation', 'innovation_eval', 'innovation_advice',
                         'alignment', 'alignment_eval', 'alignment_advice',
                         'novelty', 'novelty_eval', 'novelty_advice',
                            'filter_score', 'relevance_problem', 'clarity_problem',
                            'suitability_solution','clarity_solution']
            data_query = data_query[columns]
            data_query_T = data_query.T.reset_index()
            data_query_T.columns = ['Metric', 'Details']
            st.dataframe(data_query_T, width = 895)
        try:
            last_row = selected_rows[-1]
        except IndexError:
            return
        df["select"] = False
        st.session_state["data"] = df
        st.session_state["editor_key"] = random.randint(0, 100000)
        st.session_state["last_selected_row"] = df.iloc[last_row]

    if df_metrics_interface.shape[0] > 0:
        st.session_state["data"] = pages[current_page - 1]

        data = pagination.data_editor(
            data = st.session_state["data"], 
            use_container_width=True,
            hide_index = True,
            disabled=['id','problem','solution','category',
                        'overall_score', 'market_potential', 'feasibility', 'scalability', 'innovation', 'alignment', 'novelty',
                        'filter_score', 'relevance_problem', 'clarity_problem',
                        'suitability_solution','clarity_solution'],
            column_order= ('id','details','problem','solution','category',
                            'overall_score', 'market_potential', 'feasibility', 'scalability', 'innovation', 'alignment', 'novelty',
                            'filter_score', 'relevance_problem', 'clarity_problem',
                            'suitability_solution','clarity_solution'),
            key=st.session_state["editor_key"],
            on_change=get_row_and_clear_selection
        )
    else:
        pagination.error("Choose a lower overall score threshold and/or select at least one category.")
