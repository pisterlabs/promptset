from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from dotenv import load_dotenv
import os
import streamlit as st
import streamlit.components.v1 as components
import zipfile
from io import BytesIO

# Load environment variables
load_dotenv()

# Define a constant for the base HTML template
BASE_HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Generated Web App</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Lexend&family=Poppins&family=Space+Grotesk&display=swap" rel="stylesheet">
    <style>
        body{
            font-family: 'Space Grotesk', sans-serif;
        }
        h3 {
            color: yellow;
        }
    </style>
</head>
<body>
    <h3>Your web app will be displayed here</h3>
</body>
</html>
"""

prompt = """
You are an experienced web developer. Please generate a complete functional website using JavaScript, HTML, and CSS.
User requirements are: "{text}"
The JavaScript (JS) file and CSS file should be combined into one HTML file which can be directly run on a server.
If there is no CSS requirement specified, please add an amazing style CSS in your response. If there is no font family specified by user then user google fonts. you are using poppins either Space Grotesk.
For theme and color use this color: 
HTML:
"""

# Initialize Prompt template
WEBAPP_PROMPT = PromptTemplate(template=prompt, input_variables=["text"])
# Initialize the ChatOpenAI model
llm = ChatOpenAI(temperature=0.7)
llm_chain = LLMChain(llm=llm,prompt=WEBAPP_PROMPT,verbose=True) 

# Initialize Streamlit page configuration
st.set_page_config(layout="wide")

# Initialize the Streamlit session state with default values
if "html" not in st.session_state:
    st.session_state.html = BASE_HTML_TEMPLATE


if "requirements" not in st.session_state:
    st.session_state["requirements"] = ""
    
with st.container():
    st.title("Web App Generator")
    st.header("Predefined Examples")
    st.write("Choose from the following predefined examples or specify your requirements:")
    
    # Predefined examples with descriptions
    predefined_examples = {
        "Simple Landing Page": "Create a simple landing page with a title and a button.",
        "Simple Login Page":"Create a simple login page with username and password with remember me checkbox.",
        "Blog Website": "Build a blog website with multiple articles and navigation.",
        "E-commerce Site": "Design an e-commerce website with product listings and a shopping cart.",
        "Portfolio Website": "Create a portfolio website showcasing your work and projects.",
        "Event Registration Page": "Build an event registration page with a form for attendees."
    }
    key_list=list(predefined_examples.keys())
    val_list=list(predefined_examples.values())

    # Define a variable to store the selected example
    selected_example = None
    col_buttons = st.columns(len(predefined_examples))
    c1, c2, c3, c4, c5, c6 = st.columns(6,gap='small')
    # Display predefined examples as buttons or links
    with c1:
        if st.button(key_list[0]):
            selected_example = val_list[0]
    with c2:
        if st.button(key_list[1]):
            selected_example = val_list[1]
    with c3:
        if st.button(key_list[2]):
            selected_example = val_list[2]
    with c4:
        if st.button(key_list[3]):
            selected_example = val_list[3]
    with c5:
        if st.button(key_list[4]):
            selected_example = val_list[4]
    with c6:
        if st.button(key_list[5]):
            selected_example = val_list[5]
  
    if selected_example is not None:
        st.session_state["requirements"] = selected_example
        st.experimental_rerun()
 
# Define a Streamlit column layout
col1, col2 = st.columns([0.5, 0.5], gap='medium')



# Sidebar column (col1)
with col1:
    st.write("What kind of web app do you want to create?")
    requirements = st.text_area("Requirements",value=st.session_state["requirements"])
    if st.button("Create a website"):
        # Generate HTML content using the ChatOpenAI model
        if requirements:
            try:
                # Generate HTML content based on user requirements
                generated_html = llm_chain.run(prompt.format(text=requirements))
                st.session_state.html = generated_html
                st.session_state["requirements"] = requirements

            except Exception as e:
                st.session_state.error_message = f"An error occurred: {str(e)}"

    # Display generated HTML code with syntax highlighting
    st.code(st.session_state.html, language="html")

    # 4. Download Option
    if st.button("Download Generated Website"):
        # Create a ZIP file in memory
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zipf:
            # Add HTML content to the ZIP file
            zipf.writestr("index.html", st.session_state.html)
            # You can add CSS and JS files here if needed

        # Prepare the ZIP file for download
        zip_buffer.seek(0)
        st.download_button(
            label="Click to Download",
            data=zip_buffer,
            file_name="generated_website.zip",
            key="download_button",
        )

   


# Main content column (col2)
with col2:
    # Display the generated website using the components.html method
    components.html(st.session_state.html, height=600, scrolling=True)

# st.session_state.req = shared_req
