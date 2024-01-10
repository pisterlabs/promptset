import praw
from datetime import datetime
import pandas as pd
from openai import OpenAI
import os
import streamlit as st
st.set_page_config(layout="wide")

st.title = 'Reddit Posts and Comments'

# Load the API key from Streamlit secrets

openai_api_key = os.environ.get('openai_api_key')
client_id = os.environ.get('client_id')
client_secret = os.environ.get('client_secret')
username = os.environ.get('username')
password = os.environ.get('password')
user_agent = os.environ.get('user_agent')

#openai.api_key = st.secrets["openai"]["openai_api_key"]
#client_id = st.secrets["reddit"]["client_id"]
#client_secret = st.secrets["reddit"]["client_secret"]
#username = st.secrets["reddit"]["username"]
#password = st.secrets["reddit"]["password"]
#user_agent = st.secrets["reddit"]["user_agent"]

# Initialize the Reddit instance
reddit = praw.Reddit(client_id=client_id,
                     client_secret=client_secret,
                     username=username,
                     password=password,
                     user_agent=user_agent)


def refresh_data():
    subreddits = ["kubernetes", "devops"]  # Example subreddits
    all_data = []
    for subreddit_name in subreddits:
        flattened_submissions = get_flattened_submissions(subreddit_name)
        all_data.extend(flattened_submissions)

    st.session_state['loaded_data'] = pd.DataFrame(all_data)


@st.cache_data
def get_submissions(subreddit_name):
    subreddit = reddit.subreddit(subreddit_name)
    submissions = subreddit.new(limit=20)
    submissions_with_comments = []
    for submission in submissions:
        comments = [comment.body for comment in submission.comments.list()[:10]]  # Fetching top 10 comments
        submission_with_comments = {
            "Subreddit": subreddit_name,
            "Title": submission.title,
            "Author": str(submission.author),
            "Score": submission.score,
            "Number of Comments": submission.num_comments,
            "Timestamp": datetime.utcfromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
            "Comments": comments
        }
        submissions_with_comments.append(submission_with_comments)
    return submissions_with_comments




#@st.cache_data
def get_flattened_submissions_with_search(subreddits, search_term=None, progress_text=None):
    all_data = []
    max_posts = 20  # Maximum number of posts per subreddit
    max_comments_per_post = 10

    if not subreddits:
        # If no subreddits are provided, use 'all'
        subreddits = ['all']

    for subreddit_name in subreddits:
        if subreddit_name == 'all':
            search_results = reddit.subreddit('all').search(search_term, limit=max_posts)
        else:
            subreddit = reddit.subreddit(subreddit_name)
            search_results = subreddit.search(search_term, limit=max_posts) if search_term else subreddit.new(limit=max_posts)

        for submission in search_results:
            # Add the main post
            flattened_data = {
                "Timestamp": datetime.utcfromtimestamp(submission.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                "Subreddit": submission.subreddit.display_name,  # Actual subreddit of the post
                "Title": submission.title,
                "Post Text": submission.selftext,  # Include the text of the post
                "Comments": "Post",
                "Author": str(submission.author),
                "Score": submission.score,
                "Number of Comments": submission.num_comments  # Number of comments for the post

            }
            all_data.append(flattened_data)

            # Add comments until max_comments is reached
            total_comments = 0
            for comment in submission.comments.list():
                if total_comments >= max_comments_per_post:
                    break
                if isinstance(comment, praw.models.MoreComments):
                    continue

                total_comments += 1

                flattened_data = {
                    "Timestamp": datetime.utcfromtimestamp(comment.created_utc).strftime('%Y-%m-%d %H:%M:%S'),
                    "Subreddit": submission.subreddit.display_name,  # Actual subreddit of the comment's parent post
                    "Title": submission.title,
                    "Post Text": "",  # Leave blank for comments
                    "Comments": comment.body,
                    "Author": str(comment.author),
                    "Score": comment.score,
                    "Number of Comments": ""  # Leave blank for comments

                }
                all_data.append(flattened_data)

            if progress_text:
                progress_text.text(f"Processed posts in {subreddit_name}...")

    if progress_text:
        progress_text.empty()

    return all_data



# Initialize 'loaded_data' in session state if not present
if 'loaded_data' not in st.session_state:
    st.session_state['loaded_data'] = pd.DataFrame()

# Button to submit selections
st.sidebar.title("Subreddit Selection")
selected_subreddits = st.sidebar.multiselect("Choose Subreddits", ["kubernetes", "devops", "python", "datascience", "opentelemetry", "observability", "ebpf"])
search_term = st.sidebar.text_input("Enter a Search Term", "")

# Button to submit selections
# Button to submit selections
if st.sidebar.button("Submit"):
    if not selected_subreddits and not search_term:
        st.error("Please select at least one subreddit or enter a search term.")
    else:
        with st.spinner("Fetching and processing data..."):
            progress_text = st.empty()
            if search_term and not selected_subreddits:
                fetched_data = get_flattened_submissions_with_search(["all"], search_term, progress_text)
            else:
                fetched_data = get_flattened_submissions_with_search(selected_subreddits, search_term, progress_text)

            st.session_state['loaded_data'] = pd.DataFrame(fetched_data)
            st.success("Data loaded successfully!")
            progress_text.empty()

            # Display AgGrid with the loaded data
            st.header('Subreddit Posts and Comments')

st.dataframe(st.session_state['loaded_data'])



def extract_key_terms(text):
    """
    Function to extract key terms from the text using OpenAI.
    """
    response = openai_client.Completion.create(
        model="gpt-4-1106-preview",
        prompt=[
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": f"Extract key terms from this text and display them in a bullet list: {text}"
            }
        ],
        max_tokens=150
    )

    return response['choices'][0]['message']['content'].strip().split(', ')

def create_term_frequency_df(summaries):
    """
    Function to create a DataFrame with term frequencies for visualization.
    """
    all_terms = []
    for subreddit, summary in summaries.items():
        terms = extract_key_terms(summary)
        for term in terms:
            all_terms.append({'Term': term, 'Subreddit': subreddit, 'Frequency': 1})

    # Create a DataFrame and aggregate the frequencies
    df = pd.DataFrame(all_terms)
    return df.groupby(['Term', 'Subreddit']).sum().reset_index()





def estimate_token_count(text):
    """
    Estimate the number of tokens in a given text.
    """
    # Adjusted average token length for English
    return len(text) // 4

def truncate_text(text, max_input_tokens):
    """
    Truncate the text to fit within the maximum token count for the input.
    """
    words = text.split()
    truncated_text = ''
    for word in words:
        if estimate_token_count(truncated_text + word) > max_input_tokens:
            break
        truncated_text += word + ' '
    return truncated_text.strip()

api_call_count = 0

def summarize_text(text, max_input_tokens, max_completion_tokens=4096):
    global api_call_count  # Use the global counter

    if estimate_token_count(text) > max_input_tokens:
        text = truncate_text(text, max_input_tokens)

    response = openai_client.ChatCompletion.create(
        model="gpt-4-1106-preview",
        prompt=[
            {"role": "system", "content": "You are a helpful assistant. You focus on identifying and summarizing key themes within text."},
            {"role": "user", "content": f"Identify and summarize key topic and subtopics in the following information:\n\n{text}. Do not list individual posts but always summarize the bigger picture topics."}
        ],
        max_tokens=max_completion_tokens
    )
    api_call_count += 1  # Increment the counter after each API call
    return response['choices'][0]['message']['content']



def get_aggregated_subreddit_data(df):
    """
    Aggregate text data from multiple subreddits for summarization.
    """
    aggregated_texts = {}
    for subreddit in df['Subreddit'].unique():
        subreddit_data = df[df['Subreddit'] == subreddit]
        unique_titles_comments = subreddit_data[['Title', 'Comments','Post Text']].drop_duplicates()
        aggregated_text = '\n'.join(unique_titles_comments.apply(lambda x: f"Title: {x['Title']}\nComment: {x['Comments']}\nPostText: {x['Post Text']} ']", axis=1))
        aggregated_texts[subreddit] = aggregated_text
    return aggregated_texts

# Function to write content to a text file
def export_to_txt(filename, content):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(content)

if st.button("Summarize Subreddit Data"):
    max_input_tokens = 120000  # Allocate tokens for the input text
    max_completion_tokens = 4096  # Allocate tokens for the completion (summary)

    aggregated_texts = get_aggregated_subreddit_data(st.session_state['loaded_data'])
    summaries = {}
    key_terms = {}

    # Placeholder for API call count
    api_call_count_placeholder = st.empty()

    # Directory to store text files
    export_dir = 'exported_summaries'
    if not os.path.exists(export_dir):
        os.makedirs(export_dir)

    for subreddit, text in aggregated_texts.items():
        # Export the text to a file before summarizing
        export_filename = f"{export_dir}/{subreddit}_to_summarize.txt"
        export_to_txt(export_filename, text)
        st.text(f"Content for {subreddit} exported to {export_filename}")

        # Summarize the text
        summary = summarize_text(text, max_input_tokens, max_completion_tokens)
        summaries[subreddit] = summary
        key_terms[subreddit] = extract_key_terms(summary)

        # Update the API call count dynamically
        api_call_count_placeholder.write(f"API calls made so far: {api_call_count}")

    # Store summaries and key terms in session state
    st.session_state['summaries'] = summaries
    st.session_state['key_terms'] = key_terms

    # Final API call count
    api_call_count_placeholder.write(f"Total API calls made: {api_call_count}")

# Check if summaries and key terms are in session state and display them
if 'summaries' in st.session_state and 'key_terms' in st.session_state:
    for subreddit, summary in st.session_state['summaries'].items():
        st.subheader(f"Summary for {subreddit}")
        st.text_area(f"{subreddit} Summary", summary, height=500)
        st.text_area("Key Terms", ", ".join(st.session_state['key_terms'][subreddit]), height=300)

