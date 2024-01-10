import streamlit as st
from langchain import PromptTemplate
from langchain.llms import OpenAI
import math

# LLM loading function
def load_LLM(openai_api_key):
    llm = OpenAI(openai_api_key=openai_api_key)
    return llm


# Generate content plan with GPT-3
def generate_content_plan(llm, content_input, video_length, video_style, tone):
    prompt = f"""Generate a {content_type} {video_length}-minute long video content plan based on the article. Prioritize key insights and allocate time to each section based on its complexity, relevance to the overall topic, and fitting within the {video_length} time frame. Specify the time in seconds for each section. The article is as follows: {content_input}
    Make sure to distribute the time effectively among sections to ensure a balanced and informative video within the specified time limit.For sections, ensure that the total duration does not exceed {video_length}. the video is for {video_style} and users are {tone}
    Each section should be described as a JSON-like object: {{ "Title": "section_title", "Description": "description", "Duration": "duration in seconds" }}




    """
    content_plan = llm(prompt)
    return content_plan
# Generate detailed content for each section
def generate_detailed_content(llm, content_input, video_length, video_style, tone, content_type, content_plan):
    prompt = f"""Based on the content plan provided below, create a detailed script for a {video_length}-minute video:
    Content Plan: {content_plan}

    Please note:
    - 'Title' refers to the title of the section.
    - 'Description' refers to a brief overview of the section.
    - 'Duration' refers to the time allocated to each section in seconds.

    The script should include:
    - Introductory lines for each section
    - Key points for each section
    - Data or examples to support key points
    - Transitions between sections
    - A closing statement

    Additional Context:
    - Content Type: {content_type}
    - Video Style: {video_style}
    - Target Audience: {tone}
    - Base Article: {content_input}
    """
    
    detailed_content = llm(prompt)
    return detailed_content



# ... Your existing functions

# Streamlit UI

# Streamlit UI
st.set_page_config(page_title="ScriptMaven📜🧠", page_icon=":robot:")
st.header("Script:red[Maven] 📜🧠")
st.subheader("Your Edu-Video Blueprint", divider='rainbow')
st.write("Automate and optimize your educational video scripting with ScriptMaven. Leveraging OpenAI's GPT-4, it transforms raw text into structured video scripts tailored for various audiences and styles.")
# Add the Side Panel
st.sidebar.header('Product Description')
st.sidebar.markdown("""
**Key Features:**
- Multiple video styles and tones.
- Adjustable video lengths.
- Varied content types.
""")
st.sidebar.header("Who's it for?")
st.sidebar.markdown("""
- Content Creators
- Corporate Trainers
- Educational Institutions
""")

st.sidebar.header('How to Use:')
st.sidebar.markdown("""
[Get your OpenAI API Key from here](https://platform.openai.com/signup)
1. **API Key**: Enter OpenAI API key.
2. **Style & Tone**: Select from options.
3. **Video Length & Type**: Choose.
4. **Raw Content**: Paste.
5. **Generate Plan**: Click.
6. **Review & Finalize**: Obtain the script.
""")
# icons
# Add Font Awesome CSS
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
    """, unsafe_allow_html=True)

# ... Existing code for sidebar and Font Awesome inclusion

# Add social media icons at the bottom
st.sidebar.markdown('---')  # Horizontal line for separation
st.sidebar.markdown("### Socials:")
st.sidebar.markdown("""
<div style="display: flex; flex-direction: row; align-items: center;">
    <a href="https://github.com/pathakmukul" target="_blank" style="margin-right: 20px; text-decoration: none;">
        <i class="fab fa-github fa-2x" style="color: white;"></i>
    </a>
    <a href="https://twitter.com/twitter" target="_blank" style="margin-right: 20px; text-decoration: none;">
        <i class="fab fa-twitter fa-2x" style="color: inherit;"></i>
    </a>
    <a href="https://huggingface.co/broductmanager" target="_blank" style="font-size: 28px; text-decoration: none; color: inherit;">
        🤗
    </a>
</div>
""", unsafe_allow_html=True)

# f
# Create columns for the API Key input and the Get Key link

openai_api_key = st.text_input("OpenAI API Key", placeholder="Ex: sk-2twmA8tfCb8un4...")
# Create 2x2 grid for the select boxes
col1, col2 = st.columns(2)
with col1:
    video_style = st.selectbox('Video Style:', ('YouTube', 'Corporate'))
    video_length = st.selectbox('Video Length(minutes):', ('3','5', '10', '15'))

with col2:
    tone = st.selectbox('Audience:', ('School Students', 'College Students', 'Employee', 'For Teens'))
    content_type = st.selectbox('Content Type:', ('Case Study', 'Masterclass', 'Documentary', 'How-to Videos', 'Coding', 'Summary', 'Review'))

# Text area below the 2x2 grid
content_input = st.text_area("Content Input", placeholder="Your content deserves to be here 👑 ")


# Generate plan button
# ...
# Generate Content Plan Button
if st.button("Generate Plan"):
    llm = load_LLM(openai_api_key=openai_api_key)
    content_plan = generate_content_plan(llm, content_input, video_length, video_style, tone)

    # Display the generated plan
    st.write("Generated Content Plan:")
    st.write(content_plan)

    # Pass the generated content_plan directly into the next prompt
    detailed_content = generate_detailed_content(llm, content_plan, content_input, video_length, video_style, tone, content_type)
    
    st.write("Generated Detailed Scripts:")
    st.write(detailed_content)
    
    
# Add footer
st.markdown(
    "<div style='text-align: center;'>Made with <span style='color: red;'>&hearts;</span> by Mukul.</div>",
    unsafe_allow_html=True,
)
