import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from streamlit_chat import message
from PIL import Image
import base64
import io
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory

# Streamlit app
def image():


    def process_image(uploaded_file):
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Process the image and return the URL or other information
        # For demonstration purposes, convert the image to base64 and return a data URL
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
        image_url = f"data:image/jpeg;base64,{image_base64}"

        return image_url
    apiKey = "AIzaSyAXkkcrrUBjPEgj93tZ9azy7zcS1wI1jUA"

    llm = ChatGoogleGenerativeAI(model="gemini-pro-vision", google_api_key=apiKey)

    image_url = None  # Initialize image_url outside the if statement

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image_url = process_image(uploaded_file)


    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    prompt = st.chat_input("Say something")
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": prompt,
            },  # You can optionally provide text parts
            {"type": "image_url", "image_url": image_url},
        ]
    )

    if prompt:
        with st.chat_message("user").markdown(prompt):
            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": prompt
                }
            )
        response = llm.invoke([message])
        text_output = response.content

        with st.chat_message("assistant").markdown(text_output):
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": text_output
                }
            )


