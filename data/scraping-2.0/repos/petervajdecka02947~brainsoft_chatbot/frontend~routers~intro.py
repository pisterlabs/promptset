import os
from dotenv import load_dotenv
from PIL import Image
import streamlit as st

# Load environment variables
load_dotenv()

# Accessing the variables from .env file
PAGE_ICON = os.getenv("PAGE_ICON")
PAGE_TITLE = os.getenv("PAGE_TITLE")
PROFILE_IMAGE_PATH = os.getenv("PROFILE_IMAGE_PATH")
AUTHOR_NAME = os.getenv("AUTHOR_NAME")
AUTHOR_EMAIL = os.getenv("AUTHOR_EMAIL")


def set_page_configuration():
    """
    Configures the Streamlit page settings such as layout, page icon, and title.
    """
    st.set_page_config(layout="wide", page_icon=PAGE_ICON, page_title=PAGE_TITLE)


def display_author_info():
    """
    Displays the author's information in a sidebar expander on the Streamlit page.
    This includes an image, the author's name, and contact email.
    """
    with st.sidebar.expander("📬 Author"):
        image = Image.open(PROFILE_IMAGE_PATH)
        st.image(image)
        st.write(f"**Created by {AUTHOR_NAME}**")
        st.write(f"**Mail**: {AUTHOR_EMAIL}")


def display_main_title():
    """
    Displays the main title of the Streamlit page using HTML formatting.
    """

    st.markdown(
        """
        <h2 style='text-align: center;'>Brainy, Generative AI Python SDK documentation assistant 🤖</h2>
        """,
        unsafe_allow_html=True,
    )


def display_description():
    """
    Displays a description of the application on the Streamlit page, including its purpose and usage, with HTML formatting.
    """
    st.markdown("---")
    st.markdown(
        """ 
        <h5 style='text-align:center;'>This generative chatbot was created for Python AI Code Challenge at BrainSoft🧠</h5>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")


def display_sources_info():
    """
    Displays information about the sources and usage instructions for the Streamlit application.
    It includes steps for API token requirement, model selection, source selection, and how to chat and explore.
    """
    st.header("Getting Started")
    st.markdown(
        """
            #### Step 1: API Token Requirement
            - **First**, you'll need an **API token from OpenAI**. This token is essential for the application to access and utilize OpenAI's powerful AI models.

            #### Step 2: Model Selection
            - **Next**, choose an OpenAI model for your chatbot. We support models like **gpt-3.5-turbo-16k, gpt-4-1106-preview, and gpt-4**.
            - You can find more details about each model's capabilities, including their length, pricing, and other specifics at the [OpenAI Models Documentation](https://platform.openai.com/docs/models/continuous-model-upgrades).

            #### Step 3: Source Selection for Interaction
            - **Choose a Source**:
                - **IBM Generative SDK**: Interact with the Generative AI Python SDK documentation.
                - **Other Files**: Engage with documents you upload (in PDF, TXT, CSV formats).

            #### Step 4: Chat and Explore
            - Once the model and source are set up, start chatting with the AI. Whether it's querying the SDK or discussing your uploaded documents, the AI is ready to assist."""
    )

    st.header("Features of the Chatbot")
    st.markdown(
        """
                - **Conversation Memory**: The chatbot remembers the entire conversation history, enhancing the context and relevance of its responses.
                - **Source Flexibility**: Independent of your source choice, the chatbot streams answers efficiently and accurately.
                - **Advanced Embeddings**: Uses 'text-embedding-ada-002' embeddings from OpenAI for nuanced understanding and information retireval.
                - **Transparency in Responses**: `At this version, the chatbot cites the source related to current question not chat history, ensuring clarity and trust.`
                - **Reset Option**: A 'Reset Conversation' feature allows you to start afresh anytime, storing each session independently in MongoDB.
                - **Web Lookups**: When using the 'IBM Generative SDK' source, the chatbot can perform specific web searches for additional information.
                """
    )
    st.header("How to Use the Chatbot")
    st.markdown(
        """
    - **Interactive Chat**: Once the setup is complete, interact with the chatbot by asking questions or starting a conversation.
    - **Response Source**: For every answer, the chatbot provides the source, whether it's from the SDK documentation, your uploaded files, or web lookups.
    - **History and Reset**: View the conversation history and use the reset feature to start a new conversation thread.
    """
    )

    st.markdown("---")


def get_user_api_key():
    """
    Captures the user's OpenAI API key input from the Streamlit sidebar.

    Returns:
        str: The API key input by the user.
    """
    """Gets the user API key input."""
    return st.sidebar.text_input(
        label="#### Insert your OpenAI API key if you have one:",
        placeholder="Paste your OpenAI API key, sk-",
        type="password",
    )


def get_source_selection():
    """
    Allows the user to select the source for interaction from the Streamlit sidebar.

    Returns:
        str: The source selected by the user from the available options.
    """
    options = ["IBM Generative SDK", "Other files"]
    return st.sidebar.selectbox("Choose a source:", options)


# Main script execution
if __name__ == "__main__":
    set_page_configuration()
    display_author_info()
    display_main_title()
    display_description()
    display_sources_info()
    user_api_key = get_user_api_key()
    selected_option = get_source_selection()
