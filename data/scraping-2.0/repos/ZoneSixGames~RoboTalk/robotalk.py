# Bring in deps
from decouple import config
import os
import streamlit as st
import requests
from bs4 import BeautifulSoup
from langchain import LLMChain, OpenAI  # Import the correct class
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.utilities import GoogleSearchAPIWrapper
from elevenlabs import generate, save, voices
import urllib.parse
import feedparser
from datetime import datetime
from pydub import AudioSegment
import nltk

# Access the environment variables
API_KEYS = {
    'OPENAI_API_KEY': config('OPENAI_API_KEY'),
    'ELEVENLABS_API_KEY': config('ELEVENLABS_API_KEY'),
    'ELEVENLABS_VOICE_1_ID': config('ELEVENLABS_VOICE_1_ID'),
    'ELEVENLABS_VOICE_2_ID': config('ELEVENLABS_VOICE_2_ID'),
    'ELEVENLABS_VOICE_3_ID': config('ELEVENLABS_VOICE_3_ID'),
    'ELEVENLABS_VOICE_4_ID': config('ELEVENLABS_VOICE_4_ID'),
    'ELEVENLABS_VOICE_5_ID': config('ELEVENLABS_VOICE_5_ID'),
    'ELEVENLABS_VOICE_6_ID': config('ELEVENLABS_VOICE_6_ID'),
    'ELEVENLABS_VOICE_7_ID': config('ELEVENLABS_VOICE_7_ID'),
    'ELEVENLABS_VOICE_8_ID': config('ELEVENLABS_VOICE_8_ID'),
    'GOOGLE_CSE_ID': config('CUSTOM_SEARCH_ENGINE_ID'),
    'GOOGLE_API_KEY': config('GOOGLE_API_KEY'),
}

# Application Framework
st.title('RoboTalk Podcast Creator by Zone Six')

# Collect the inputs
prompt = st.text_input("Enter the podcast topic")
p1_name = st.text_input("Host Name")
p1 = st.text_input("Enter the personality for the Host")

# Initialize environment
os.environ.update(API_KEYS)

# Initialize components
google_search_tool = GoogleSearchAPIWrapper()

# Initialize OpenAI API
openai_llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k")  # Initialize the OpenAI LLM

# Define templates
title = PromptTemplate.from_template("Write a witty, funny, or ironic podcast title about {topic}.")
script = PromptTemplate.from_template("Write a first person editorial podcast based on a given title, research, and unique author personality. Title: {title}, Research: {news_research}, Personality: {p1_name}: {p1}. The article should start by giving an introduction to the topic and then offering an opinion based on the personality of the author. Do not use formal words like 'in conclusion' or 'however' or 'furthermore'.")
# cont_script = PromptTemplate.from_template("Continue writing a podcast script based on a given title, research, recent podcast discussion history. Title: {title}, Research: {research}, Script: {script}")
news = PromptTemplate.from_template("Summarize this news story: {story}")
research = PromptTemplate.from_template("Summarize the research into talking points: {research}")

# Initialize chains
chains = {
    'title': LLMChain(llm=openai_llm, prompt=title, verbose=True, output_key='title'),
    'script': LLMChain(llm=openai_llm, prompt=script, verbose=True, output_key='script'),
    # 'cont_script': LLMChain(llm=openai_llm, prompt=cont_script, verbose=True, output_key='cont_script'),
    'news': LLMChain(llm=openai_llm, prompt=news, verbose=True, output_key='summary'),
    'research': LLMChain(llm=openai_llm, prompt=research, verbose=True, output_key='research'),
}

# Initialize session state for script, research, title if they don't exist
if 'script' not in st.session_state:
    st.session_state.script = "Script will appear here"

if 'title' not in st.session_state:
    st.session_state.title = "Podcast Title Will Appear Here"

if 'news' not in st.session_state:
    st.session_state.news = ""

if 'research' not in st.session_state:
    st.session_state.research = ""

if 'podcast_dir' not in st.session_state:
    st.session_state.podcast_dir = ""

def extract_news_text(url):
    """Extract the text of a news story given its URL."""
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')

    # Concatenate all paragraphs into a single string
    story_text = ' '.join([p.get_text() for p in paragraphs])

    # Tokenize the story_text
    tokens = nltk.word_tokenize(story_text)

    # Only keep the first XXXX tokens
    tokens = tokens[:2800]

    # Rejoin the tokens into a single string
    story_text = ' '.join(tokens)

    return story_text

def get_top_news_stories(topic, num_stories=5):
    """Get the top num_stories news stories on the given topic."""
    # URL encode the topic to ensure it's valid in a URL
    topic = urllib.parse.quote_plus(topic)
    # Get the feed from the Google News RSS
    feed = feedparser.parse(f'https://news.google.com/rss/search?q={topic}')

    # Return the top num_stories stories
    return feed.entries[:num_stories]

def summarize_news_stories(stories):
    """Summarize each news story using the OpenAI model."""
    summaries = []
    total_tokens = 0
    for story in stories:
        # Extract the URL from the story metadata
        url = story.get('link', '')
        if url:
            # Extract the news text
            story_text = extract_news_text(url)

            # Generate a summary
            summary = chains['news'].run(story_text)

            # Add summary to the list if it doesn't exceed the token limit
            summary_tokens = len(summary.split())  # rough token count
            if total_tokens + summary_tokens <= 10000:
                summaries.append(summary)
                total_tokens += summary_tokens
            else:
                break  # stop if we reach the token limit
    return summaries

def validate_inputs(prompt, p1, p1_name):
    return all([prompt, p1, p1_name])

def create_podcast_directory():
    now = datetime.now()  # get current date and time
    date_time = now.strftime("%Y_%m_%d_%H_%M_%S")  # format as a string
    podcast_dir = f"Podcast_{date_time}"  # prepend "Podcast_" to the string

    if not os.path.exists(podcast_dir):
        os.makedirs(podcast_dir)

    return podcast_dir

def convert_script_to_audio(script_text, podcast_dir):
    selected_voice_id = API_KEYS.get(voice_options[selected_voice])
    print(selected_voice_id)  # Add this line to check the selected voice ID

    if selected_voice_id is None:
        st.error("Selected voice not found.")
        return []

    audio = generate(text=script_text, api_key=API_KEYS['ELEVENLABS_API_KEY'], voice=selected_voice_id)
    audio_file = f"{podcast_dir}/podcast.mp3"  # Save in podcast directory
    save(audio=audio, filename=audio_file)
    print(audio_file)  # Add this line to check the audio file path
    return [audio_file]  # Return a list with one audio file

# Operational Structure
if st.button('Research') and validate_inputs(prompt, p1, p1_name):
    # Research and summarize top news stories
    stories = get_top_news_stories(prompt)
    news_summaries = summarize_news_stories(stories)
    research_summary = chains['research'].run(research=' '.join(news_summaries))  # Use the research chain
    st.session_state.research = research_summary  # Store the research summary in the session state
    st.session_state.podcast_dir = create_podcast_directory()
    with open(f"{st.session_state.podcast_dir}/podcast_research.txt", 'w') as f:
        f.write(st.session_state.research)
    st.success(f"Research saved in {st.session_state.podcast_dir}/podcast_research.txt")

if st.button('Generate Script') and validate_inputs(prompt, p1, p1_name):
    # Generate title
    title_result = chains['title'].run(topic=prompt)
    st.session_state.title = title_result

    # Generate and display initial script
    script_result = chains['script'].run(
        title=st.session_state.title,
        news_research=st.session_state.research,  # Use the research summary
        p1_name=p1_name,
        p1=p1,
    )
    st.session_state.script = script_result

    # Display and edit the script
    edited_script = st.text_area('Edit the Script', st.session_state.script, key='edit_script', height=300)

    # Check if the script has been modified
    if edited_script != st.session_state.script:
        st.session_state.script = edited_script

# Save the edited script to the session state and to a text file
if st.button('Save Script') and 'edit_script' in st.session_state:
    edited_script = st.session_state.edit_script

    # Update the session state with the edited script
    st.session_state.script = edited_script

    # Save the edited script to the text file
    with open(f"{st.session_state.podcast_dir}/podcast_script.txt", 'w') as f:
        f.write(edited_script)
    st.success(f"Edited script saved in {st.session_state.podcast_dir}/podcast_script.txt")
    # Display the script from the session state
    st.write(f'Script: \n{st.session_state.script}')
    
# Define the available voice options
voice_options = {
    'Voice 1': 'ELEVENLABS_VOICE_1_ID',
    'Voice 2': 'ELEVENLABS_VOICE_2_ID',
    'Voice 3': 'ELEVENLABS_VOICE_3_ID',
    'Voice 4': 'ELEVENLABS_VOICE_4_ID',
    'Voice 5': 'ELEVENLABS_VOICE_5_ID',
    'Voice 6': 'ELEVENLABS_VOICE_6_ID',
    'Voice 7': 'ELEVENLABS_VOICE_7_ID',
    'Voice 8': 'ELEVENLABS_VOICE_8_ID',
}

# Allow the user to choose a voice
selected_voice = st.selectbox("Select a voice", list(voice_options.keys()))

if st.button('Create Podcast') and st.session_state.script:
    audio_files = convert_script_to_audio(st.session_state.script, st.session_state.podcast_dir)
    if audio_files:
        st.audio(audio_files[0], format='audio/mp3')  # Use audio_files directly

with st.expander('News Summaries'):
    st.write(st.session_state.research)

with st.expander('Script'):
    st.write(st.session_state.title)
    st.write(st.session_state.script)
