import streamlit as st
import cohere 
from utils import get_links, detect_text, get_info_for_tenant, set_tenant_in_flask
from dotenv import load_dotenv
import os
load_dotenv()  # loads variables from .env

cohereAPIKey = os.getenv("COHERE_API_KEY")
co = cohere.Client(cohereAPIKey)

if "tenant_name" not in st.session_state:
    st.session_state.tenant_name = ""

if "form_object_id" not in st.session_state:
    st.session_state.form_object_id = ""


user_id = st.session_state.tenant_name  # Retrieve tenant name
response = st.session_state.form_object_id

#st.session_state['tenant_name'] = user_id  
#st.session_state["form_object_id"] = response

if user_id:
    #set_tenant_in_flask(user_id)  # Set tenant name in Flask session
    try:
        form_responses = get_info_for_tenant(user_id, response)
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.error("Tenant ID is not set. Please login first.")

def show_chat():
    st.title("BYTE Chatbot (Using cohere chat endpoint & RAG) ")
    st.write("This is a chatbot")

def display_links(links):
    # Sidebar for product links
    with st.sidebar:
        # Create a container
        add_container = st.container()
        # Use the container as a context manager to add content
        with add_container:
            st.subheader('Products recommended just for you', divider='rainbow')
            for link in links:
                st.markdown(f"* [{link}]({link})", unsafe_allow_html=True)


# applying styles.css
def load_css():
    with open("./static/styles.css", "r")  as f:
        css = f"<style>{f.read()} </style>"
        st.markdown(css, unsafe_allow_html = True)

def extract_links(documents):
    links = [doc['url'] for doc in documents if 'url' in doc]
    return ', '.join(links)

def initialize_session_state() :
    
    # Initialize a session state to track whether the initial message has been sent
    if "initial_message_sent" not in st.session_state:
        st.session_state.initial_message_sent = False

    # Initialize a session state to store the input field value
    if "input_value" not in st.session_state:
        st.session_state.input_value = ""

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
        prompt = f"""As an expert in food and nutrition, your task is to analyse my health-related information and then 
                    answer my questions related to food and nutrition keeping in mind my health specifications and goals. 
                    If I attached text extracted from my food item's nutrition label, analyse it according to my health data and
                    let me know in if it's good or bad for me, if I should or shouldn't eat it and if I 
                    can eat it, then specify the quantity I should consume as well all according to my info, all according to my
                    health information and goals as well as internet sources etc, in a brief and straightforward manner. 
                    Be concise and precise in your answers . Give clear, unambiguous answers. 
                    Your answers should be backed by verified sources from the internet. 
                    If I ask something unrelated to food and/or nutrition, politely decline. 
                    Here is my info: {form_responses} You are to give me personalized keep it according to my condition, history and goals right now 
                    I also have further information of mine that you can search by generating relavent queries to be searched for whether the food further aligns with detailed history about me.  
                    """

        st.session_state.chat_history.append({"role": "User", "message": prompt})
        st.session_state.chat_history.append({"role": "Chatbot", "message": "Yes understood, I will act accordingly and follow your instructions fully. I will also use verified internet sources for all my responses."})

def on_click_callback():

    load_css()
    customer_prompt = st.session_state.customer_prompt
    extracted_text = ""
    nutr_label = False

    # Check if an image has been uploaded
    uploaded_file = st.session_state.get("image_uploader")
    if uploaded_file is not None:
        with st.spinner('Processing image...'):
            # Directly pass the uploaded file to the detect_text function
            extracted_text = detect_text(uploaded_file)
            if extracted_text:
                # Set the extracted text as the customer prompt
                customer_prompt += extracted_text
                nutr_label = True
            else:
                st.error("No text could be extracted from the image.")

    if customer_prompt:
        
        st.session_state.input_value = ""
        st.session_state.initial_message_sent = True

        with st.spinner('Generating response...'):  

            llm_response = co.chat( 
                message=customer_prompt,
                connectors=[{"id": "web-search"}],
                documents=[],
                model='command',
                temperature=0.5,
                # return_prompt=True,
                chat_history=st.session_state.chat_history,
                prompt_truncation = 'auto',
                #stream=True,
            ) 
        llm_response_documents = llm_response.documents 
        all_links = extract_links(llm_response_documents)
        formatted_response = f"{llm_response.text}\n\nCitations:\n{all_links}"     
        st.session_state.chat_history.append({"role": "User", "message": customer_prompt})
        st.session_state.chat_history.append({"role": "Chatbot", "message": formatted_response})
        if nutr_label:
            links = get_links(customer_prompt, form_responses)
            display_links(links)
        nutr_label = False
        uploaded_file = None

def submain():
    chat_placeholder = st.container()
    prompt_placeholder = st.form("chat-form")
    initialize_session_state()



    with chat_placeholder:
        for chat in st.session_state.chat_history[2:]:
            msg = chat["message"]
            # Use emojis for user and chatbot icons
            user_icon = "🐞" if chat["role"] == "User" else "🍏"

            div = f"""
            <div class="chatRow {'rowReverse' if chat["role"] == 'User' else ''}">
                <span class="chatIcon">{user_icon}</span>
                <div class="chatBubble {'humanBubble' if chat["role"] == 'User' else 'adminBubble'}">{msg}</div>
            </div>
            """
            st.markdown(div, unsafe_allow_html=True)
            
    
    
    with st.form(key="chat_form"):
        cols = st.columns((5, 1, 3))  # Add another column for the button
        
        # Display the initial message if it hasn't been sent yet
        if not st.session_state.initial_message_sent:
            cols[0].text_input(
                "Chat",
                placeholder="Hello, how can I assist you?",
                label_visibility="collapsed",
                key="customer_prompt",
            )  
        else:
            cols[0].text_input(
                "Chat",
                value=st.session_state.input_value,
                label_visibility="collapsed",
                key="customer_prompt",
            )

        cols[1].form_submit_button(
            "Ask",
            type="secondary",
            on_click=on_click_callback,
        )

        # File uploader for images
        uploaded_file = cols[2].file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], key="image_uploader")
   
submain()
