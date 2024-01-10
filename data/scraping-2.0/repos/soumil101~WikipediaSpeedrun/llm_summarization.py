import streamlit as st
import wikipedia
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from PIL import Image
from io import BytesIO
import openai
from wikipedia.exceptions import DisambiguationError
import pandas as pd

from dotenv import load_dotenv
import os

load_dotenv()

# Set your OpenAI API key here
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize session state
if 'leaderboard' not in st.session_state:
    st.session_state.leaderboard = []

# Function to capture a screenshot of a Wikipedia page
def capture_page_screenshot(title):
    options = Options()
    options.headless = True  # Run Chrome in headless mode (no GUI)
    driver = webdriver.Chrome(options=options)
    try:
        url = wikipedia.page(title).url
        driver.get(url)
        screenshot = driver.get_screenshot_as_png()
        return Image.open(BytesIO(screenshot))
    except Exception as e:
        return None
    finally:
        driver.quit()

# Function to generate a summary using ChatGPT with specific instructions
def generate_summary(text, max_tokens=150):
    prompt = f"You are to provide help summaries in this context: the user is participating in a 'Wikipedia speedrun', a game where the user must navigate from one Wikipedia page to another using only the links within each page. Provide a brief summary containing information that summarizes each page and might help them complete the speedrun faster. {text}"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=max_tokens,  # Adjust the length of the summary as needed
        n=1,
        stop=None,
        temperature=0.7
    )
    summary = response.choices[0].text
    if len(summary) < max_tokens:
        return summary
    else:
        # If the summary is too long, reduce max_tokens and try again
        return generate_summary(text, max_tokens=max_tokens + 10)

# Function to generate a random Wikipedia page that is not the "Wing Tech" page
def generate_random_page():
    while True:
        title = wikipedia.random()
        if title != "Wing Tech":  # Exclude the problematic page
            return title

# Function to generate two random Wikipedia pages while handling DisambiguationError
def generate_random_pages():
    page_titles = []
    page_urls = []
    
    while len(page_titles) < 2:
        try:
            title = generate_random_page()
            page_titles.append(title)
            page_urls.append(wikipedia.page(title).url)
        except DisambiguationError as e:
            continue
    
    return page_titles, page_urls

# Main content
st.title("Wikipedia Speedrun")
st.write("In this game, you can generate random Wikipedia pages, capture screenshots, and generate summaries.")

# Button to generate random Wikipedia pages, capture screenshots, and generate summaries
if st.button("Generate Random Pages, Capture Screenshots, and Generate Summaries"):
    # Initialize variables to hold page titles and URLs
    page_titles, page_urls = generate_random_pages()

    # Create two columns for displaying the pages side by side
    col1, col2 = st.columns(2)

    # Fetch and display the Wikipedia pages along with screenshots and summaries
    for i, title in enumerate(page_titles):
        page_url = wikipedia.page(title).url

        # Capture a screenshot of the page
        screenshot = capture_page_screenshot(title)

        # Generate a summary using ChatGPT
        summary = generate_summary(f"Summarize the Wikipedia page about {title}. {page_url}")

        # Display the screenshot if available, or a placeholder
        if screenshot:
            with col1 if i % 2 == 0 else col2:
                st.image(screenshot, use_column_width=True, caption="Page Screenshot")
        else:
            st.warning("Failed to capture the page screenshot.")

        # Display the page title as a link
        with col1 if i % 2 == 0 else col2:
            st.subheader(f"[{title}]({page_url})")
            st.write(summary)

    # Conditional block to show the expander after generating the Wikipedia pages
    with st.expander("Add Entry to Leaderboard"):
        name = st.text_input("Name")
        start_page = st.text_input(label="Starting Page", value=page_urls[0])
        end_page = st.text_input(label="Starting Page", value=page_urls[1])
        time = st.number_input("Time (in seconds)")

        if st.button("Add to Leaderboard"):
            if name and start_page and end_page and time:
                st.session_state.leaderboard.append({"Name": name, "Start Page Link": start_page, "End Page Link": end_page, "Time (s)": time})

# Display the leaderboard
st.title("Leaderboard")
leaderboard_df = pd.DataFrame(st.session_state.leaderboard)
st.dataframe(leaderboard_df)
