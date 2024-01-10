import streamlit as st
import vertexai
import os
import openai
from google.oauth2 import service_account
from vertexai.vision_models import ImageTextModel, Image
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from gtts import gTTS
from io import BytesIO
from PIL import Image as PILImage
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import io
from streamlit_option_menu import option_menu
st.set_page_config(layout="wide")

# Replace with your Google Cloud Project ID and Location
PROJECT_ID = 'see-to-hear'
LOCATION = 'us-central1'

# Replace with your OpenAI API key
openai.api_key = "OPEN_AI_KEY"

# Replace with the path to your service account key JSON file
credentials_path = os.path.expanduser("~/Path-to-Service-Account")

# Load the credentials from the JSON file
credentials = service_account.Credentials.from_service_account_file(credentials_path)

# Initialize Vertex AI with the loaded credentials
vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)

# Load ImageTextModel from Vertex AI
model = ImageTextModel.from_pretrained("imagetext@001")

# Replace with your Azure Computer Vision subscription key and endpoint
azure_subscription_key = "AZURE_SUBSCRIPTION_KEY"
azure_endpoint = "https://heartosee.cognitiveservices.azure.com/"
computervision_client = ComputerVisionClient(azure_endpoint, CognitiveServicesCredentials(azure_subscription_key))

# Function to process uploaded image and generate captions
def gen(filegiven):
    # Read the uploaded image as bytes
    image_bytes = filegiven.read()

    # Convert bytes to a PIL Image
    source_image = PILImage.open(io.BytesIO(image_bytes))

    # Save the uploaded file to a temporary location
    with open("temp_image.jpg", "wb") as f:
        f.write(image_bytes)

    # Load the image from the saved file using the Vertex AI library
    source_image_vertexai = Image.load_from_file("temp_image.jpg")
    captions_vertexai = model.get_captions(image=source_image_vertexai)

    azure_description = computervision_client.describe_image_in_stream(io.BytesIO(image_bytes))

    combined_descriptions = [caption.text for caption in azure_description.captions]
    combined_descriptions.extend(captions_vertexai)

    openai_input = "\n".join(f"Image description: {caption}" for caption in combined_descriptions)

    # Use v1/chat/completions for chat models
    enhanced_caption = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Given these two descriptions of one image and combine them to a response less than 3 sentences while describing a scene as you were describing it to a blind person without adding information: '{combined_descriptions}'"},
        ],
    )

    if 'choices' in enhanced_caption:
        expanded_text = enhanced_caption['choices'][0]['message']['content']

        st.subheader("Enhanced Caption:")
        st.write(expanded_text)
        words = expanded_text.split()

        # Convert the improved text into speech and play the audio
        sound_file = BytesIO()
        tts = gTTS(" ".join(words), lang='en')
        tts.write_to_fp(sound_file)
        st.audio(sound_file)
    else:
        st.write("No enhanced caption available.")

# Sidebar menu with options
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Image Capture", "Image Upload", "Contact"],
        icons=["house", "camera", "cloud-arrow-up", "envelope"],
        menu_icon="cast",
        default_index=0,
    )

# Home Section
if selected == "Home":
    # Customizing the background
    page_bg_img = """
    <style>
    [data-testid="stAppViewContainer"] {
    background-color: #000000;
    opacity: 1;
    background-image:  radial-gradient(#ffffff 0.9500000000000001px, transparent 0.9500000000000001px), radial-gradient(#ffffff 0.9500000000000001px, #000000 0.9500000000000001px);
    background-size: 38px 38px;
    background-position: 0 0,19px 19px;
        }
    </style>
    """
    st.markdown("", unsafe_allow_html=True)

    # Header
    st.markdown("<h2 style='text-align: center; color: orangered;'> 👂Hear-2-See 👀</h2>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center; color: white;'> 🌟 Bridging the Gap Between Sight and Sound 🌟</h4>", unsafe_allow_html=True)
    st.divider()

    # Project Description
    with st.expander(" 'Unlocking a world of possibilities for the blind 🌎🔊🖼️'", expanded=True):
        st.markdown(
                """
                Hear2See is a project that transforms images into meaningful sounds, empowering the visually impaired to understand their surroundings like never before."
        
                **How does it work?🤔**
        
                1. Hear2See is an innovative assistive technology project designed to empower blind individuals by providing them with a unique way to comprehend their surroundings through the sense of sound. 
                
                2. This system employs image recognition technology and audio synthesis to create a seamless connection between what is seen and what is heard.
                
                📸 **Image Capture or Upload:** Users can capture images in real-time using their smartphone's camera or upload existing photos from their personal collections.
        
                🌐 **Real-Time Feedback:** Users receive instant audio feedback, enabling them to navigate their surroundings more effectively, interact with their environment, and gain a deeper understanding of the world around them.
                
                🤖 AI Learning: The system continues to learn and adapt to the user's preferences and feedback, enhancing the overall experience over time.

                🌟 Empowerment: Hear2See is not just a tool; it's a bridge to independence and inclusivity for the visually impaired. It empowers them to make informed decisions, enjoy richer experiences, and engage more fully in their daily lives.

                Hear2See is breaking down barriers and promoting a world where everyone, regardless of their visual abilities, can experience the world with greater confidence and understanding. Join us on this incredible journey of sensory transformation! 🌎🔊🖼️ 

            """
            )
   
if selected == "Image Capture":
    st.header("📸 Image Capture")
    st.write("Capture the moment with real-time photos using your smartphone's camera.")
    picture = st.camera_input("Take a picture")
    if picture is not None:
       gen(picture)

if selected == "Image Upload":
    st.header("📁 Image Upload‎")
    st.write("Select and share photos from your personal collection.")
    uploaded_file = st.file_uploader("Choose a file less than 4MB", type=("png", "jpg", "jpeg", "heic"))
    if uploaded_file is not None:
       st.image(uploaded_file)
       gen(uploaded_file)
       
def sendEmail(email, subjectentered, body):
     # Email configuration
     sender_email = 'hear2see.sender@gmail.com'
     sender_password = 'GMAIL PASSWORD'
     recipient_email = 'hear2see.recep@gmail.com'
     subject = subjectentered
     if email == "": 
         message = body + "\n Contact Information: Anonymous"
     else:
         message = body + "\n Contact Information: Anonymous" + email
     
     # Create a message
     msg = MIMEMultipart()
     msg['From'] = sender_email
     msg['To'] = recipient_email
     msg['Subject'] = subject
     msg.attach(MIMEText(message, 'plain'))
 
     # Connect to the Gmail SMTP server and send the email
     try:
         server = smtplib.SMTP_SSL('smtp.gmail.com', 465)  # Use SMTP_SSL for secure connection
         server.login(sender_email, sender_password)
         server.sendmail(sender_email, recipient_email, msg.as_string())
         server.quit()
         st.success('Email Sent Successfully!', icon="✅")
     except Exception as e:
         print(f'Error: {e}')
            
       
if selected == "Contact":
    st.header(":mailbox: Get In Touch With Us!")
    email = st.text_input("Enter Your Email Or Enter Nothing To Submit Anonymously")
    subjectentered = st.text_input("Enter Email Subject")
    txt = st.text_area("Enter Email Body")
    if st.button("Send Email"):
        sendEmail(email, subjectentered, txt)
        
        
        
