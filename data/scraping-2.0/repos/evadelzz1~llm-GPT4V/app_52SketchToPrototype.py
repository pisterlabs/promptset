from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st
import base64

if not load_dotenv():
    print("Could not load .env file or it is empty. Please check if it exists and is readable.")
    exit(1)

# Function to encode the image to base64
def encode_image(image_file):
    return base64.b64encode(image_file.getvalue()).decode("utf-8")

def extract_code_blocks(markdown_text):
    """
    This function extracts code blocks from a given markdown text.
    Code blocks are identified by lines that start and end with triple backticks
    """
    code_blocks = []
    inside_code_block = False
    current_code_block = []
    
    for line in markdown_text.split('\n'):
        if line.strip().startswith('```'):
            if inside_code_block:
                # End of current code block
                code_blocks.append('\n'.join(current_code_block))
                current_code_block = []
                inside_code_block = False
            else:
                # Start of a new code block
                inside_code_block = True
                # Optional: skip the line with backticks.
        elif inside_code_block:
            # Inside a code block, so append the line to the current code block
            current_code_block.append(line)
    
    return code_blocks if not current_code_block else code_blocks + ['\n'.join(current_code_block)]
            
# Usage in Streamlit
def show_code_in_streamlit(markdown_text):
    code_blocks = extract_code_blocks(markdown_text)
    # Convert the list of code blocks into a single HTML String
    code_extracted = '<br>'.join(f'<pre><code>{block}</code></pre>' for block in code_blocks)
    # Display the code blocks using Streamlit's components.html
    st.components.v1.html(code_extracted, height=600, width=700, scrolling = True)


st.set_page_config(page_title="Sketch-To-Prototype", layout="centered", initial_sidebar_state="collapsed")
# Streamlit page setup
st.title("Sketch To Prototype")


# Initialize the OpenAI client with the API key
# client = OpenAI(api_key=api_key)
client = OpenAI()

# File uploader allows user to add their own image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
  
# Button to trigger the analysis
analyze_button = st.button("Create the app", type="secondary")

# display uploaded images
if uploaded_file:
    # Display the uploaded image
    with st.expander("Image", expanded = True):
        st.image(uploaded_file, caption=uploaded_file.name, use_column_width=True)

# Check if an image has been uploaded, if the API key is available, and if the button has been pressed
if uploaded_file is not None and analyze_button:

    with st.spinner("Analysing your sketch ..."):
        # Encode the image
        base64_image = encode_image(uploaded_file)
        st.markdown("--------")
        
        # Optimized prompt for additional clarity and detail
        prompt_text = (
            "You are an expert tailwind developer. "
            "A user will provide you with a low-fidelity wireframe of an application and you will return a single html file that uses tailwind to create the website."
            "Use creative license to make the application more fleshed out."
            "Respond only with the html file."
        )
    
        # Create the payload for the completion request
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                    },
                ],
            }
        ]
    
        # Make the request to the OpenAI API
        try:
            # Stream the response
            with st.expander("Here's how I would code the Prototype", expanded=True):
                full_response = ""
                message_placeholder = st.empty()
                for completion in client.chat.completions.create(
                    model="gpt-4-vision-preview", 
                    messages=messages, 
                    max_tokens=1200, 
                    stream=True
                ):
                    # Check if there is content to display
                    if completion.choices[0].delta.content is not None:
                        full_response += completion.choices[0].delta.content
                        message_placeholder.markdown(full_response + "▌")
                        
                # # Final update to placeholder after the stream ends
                # message_placeholder.markdown(full_response)
            show_code_in_streamlit(full_response)
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    # Warnings for user action required
    if not uploaded_file and analyze_button:
        st.warning("Please upload an image.")
        