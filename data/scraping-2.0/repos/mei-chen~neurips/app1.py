import streamlit as st
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
import os
from openai import OpenAI
from dotenv import load_dotenv
import altair as alt

# Load environment variables
load_dotenv()

# Access your API key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

st.set_page_config(layout="wide")
st.title("NeurIPS Poster Search")

# Load CSV files
embeddings_df = pd.read_csv('embedding/embedded_combined2.csv')

# Assuming 'ada_embedding' is a string representation of a list
# Convert it to numpy array
def convert_to_numpy(embedding_str):
    return np.array(embedding_str.strip('[]').split(', '), dtype=float)

embeddings_df['ada_embedding'] = embeddings_df['ada_embedding'].apply(convert_to_numpy)

# Function to get embedding from OpenAI
def get_embedding(text, model="text-embedding-ada-002"):
    response = client.embeddings.create(input=[text], model=model)
    return np.array(response.data[0].embedding)

def google_search_url(title):
    return f"https://www.google.com/search?q={'+'.join(title.split())}"

# Function to create a Markdown table with clickable links
def create_markdown_table(df):
    # Start with the header
    markdown = "| Title | Relevance Score | Poster Session | Poster Number |\n| --- | --- | --- | --- |\n"
    
    # Add each row
    for _, row in df.iterrows():
        google_url = google_search_url(row['Title'])
        markdown += f"| [{row['Title']}]({google_url}) | {row['Relevance Score']:.2f} | {row['Poster Session']} | {row['Poster Number']} |\n"
    
    return markdown


# UMAP Visualization
df_umap = pd.read_csv('visual/umap_emb.csv')

# User input for search
query = st.text_input("What research are you interested in?", 
                      help="You can write in full sentences")


sesh = {
    1: 'Tues 10:45',
    2: 'Tues 15:15',
    3: 'Wed 10:45',
    4: 'Wed 17:00',
    5: 'Thurs 10:45',
    6: 'Thurs 17:00',
}


# Search functionality
if query:
    query_embedding = get_embedding(query)

    # Calculate cosine similarity with each embedding in the list
    similarity_scores = [1 - cosine(query_embedding, emb) for emb in embeddings_df['ada_embedding']]

    # Combine scores with titles and their indices from your dataset
    scored_titles = list(zip(similarity_scores, embeddings_df['title'], range(len(similarity_scores))))
    scored_titles.sort(key=lambda x: x[0], reverse=True)


    # Get indices of top 6 relevant papers
    top_indices = [idx for _, _, idx in scored_titles[:6]]

    # Create a copy of df_umap and add a new column for highlighting
    df_umap_copy = df_umap.copy()
    df_umap_copy['highlight'] = df_umap_copy.index.isin(top_indices)


        # Create and customize the chart with conditional color
    chart = alt.Chart(df_umap_copy).mark_point(size =111, filled=True).encode(
        x=alt.X('x', axis=alt.Axis(title=None)),
        y=alt.Y('y', axis=alt.Axis(title=None)),
        color=alt.condition(
            alt.datum.highlight,  # If the 'highlight' field is True
            alt.value('green'),    # Use red color
            alt.value('lightgray')    # Else, use blue color
        ),
        tooltip=['title']
    ).properties(
        width=600,  # Adjust the width as needed
        height=800  # Adjust the height as needed
    ).configure_axis(
        grid=False,
        labels=False,
        ticks=False,
        domain=False
    )
    

    st.altair_chart(chart, use_container_width=True)

    # Display top 6 relevant titles
    # st.write('Top 6 Relevant Papers:')
    # for score, title, _ in scored_titles[:6]:
    #     # st.write(f"{title} (Score: {score})")
    #     st.markdown(f"- **{title}** (Similarity Score: {score:.2f})")

    ########### TABLE PRINT ################

    # Display top 6 relevant titles
    st.markdown("### Top 6 Relevant Titles")

    # Retrieve session numbers for the top titles
    top_indices = [idx for _, _, idx in scored_titles[:6]]
    top_sessions = embeddings_df.iloc[top_indices]['session']
    top_poster_numbers = embeddings_df.iloc[top_indices]['poster_number']

    # Create a DataFrame for displaying
    top_titles_df = pd.DataFrame({
        'Title': [title for _, title, _ in scored_titles[:6]],
        'Relevance Score': [score for score, _, _ in scored_titles[:6]],
        'Poster Session': top_sessions.values, 
        'Poster Number': top_poster_numbers.values
    })

    # Create the Markdown table
    table_md = create_markdown_table(top_titles_df)

    # Display the table in Streamlit
    st.markdown(table_md, unsafe_allow_html=True)



    ###################### Footer ######################
    footer_note = """
        <style>
        .footer_note {
        font-size: 10px; 
        font-style: italic; 
        text-align: right;  /* Right align text */
        margin-right: 3px; /* Optional: adds some margin to the right */
        padding: 3px;      /* Optional: adds some padding around the text */
        }
        </style>
        <div class="footer_note">
        <br><br>
        <p>Special thanks to Jay Alammar</p>
        <p>Made with ❤️ by Mei Chen</p>
        <p>Feedback and suggestions --> <a href="mailto:meichen.mlc@gmail.com">meichen.mlc@gmail.com</a></p>
        </div>
        """

    st.markdown(footer_note, unsafe_allow_html=True)



# Create and customize the chart
# chart = alt.Chart(df_umap).mark_point().encode(
#     x=alt.X('x', axis=alt.Axis(title=None)),
#     y=alt.Y('y', axis=alt.Axis(title=None)),
#     tooltip=['title']
# ).configure_axis(
#     grid=False,
#     labels=False,
#     ticks=False,
#     domain=False
# )

# st.altair_chart(chart, use_container_width=True)

# # Sidebar for user input
# with st.sidebar:
#     st.write('Special thanks to Jay Alammar for helping with the UMAP visualization')
#     st.write('If you have questions or have feedback, please email me at: meichen.mlc@gmail.com')
#     st.write('My twitter is: @meigustas')


