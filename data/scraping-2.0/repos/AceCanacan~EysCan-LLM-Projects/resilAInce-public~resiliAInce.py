import streamlit as st
import openai
from streamlit_chat import message
import sys

openai.api_key = "YOUR API KEY"

# Set page title and header
st.set_page_config(page_title="resiliAInce")

################################################################################################

def gratitude_page(st):

    # Initialise session state variables
    if 'gratitude_generated' not in st.session_state:
        st.session_state['gratitude_generated'] = []
    if 'gratitude_past' not in st.session_state:
        st.session_state['gratitude_past'] = []
    if 'gratitude_messages' not in st.session_state:
        st.session_state['gratitude_messages'] = []
    if 'gratitude_model_name' not in st.session_state:
        st.session_state['gratitude_model_name'] = []

    if 'context_provided' not in st.session_state:
        st.session_state['context_provided'] = False
        st.session_state['background'] = ''
        st.session_state['reason'] = ''
        st.session_state['expectation'] = ''

    # Generate a response
    def generate_response(prompt):
        st.session_state['gratitude_messages'].append({"role": "user", "content": prompt})

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=st.session_state['gratitude_messages'],
            max_tokens=3000,
            temperature=0.6
        )
        response = completion.choices[0].message.content.strip()
        st.session_state['gratitude_messages'].append({"role": "system", "content": response})

        return response
    
    st.markdown('<h1 style="color: #fd79a8; font-weight: bold; text-align: center;">🌱 Jamie the Grounded 🌱</h1>', unsafe_allow_html=True)
    st.markdown('')


    if not st.session_state['context_provided']:
        with st.form(key='context_form'):
            st.write('Before you begin, please fill out these forms to help Jamie understand you better and provide more meaningful support based on your current situation.')
            st.write('All the information you provide in the forms will be deleted after you finish your session or refresh the page.')

            st.session_state['background'] = st.text_area("What is your background?", value=st.session_state['background'])
            st.session_state['reason'] = st.text_area("What brought you here?", value=st.session_state['reason'])
            st.session_state['expectation'] = st.text_area("What do you expect with your conversation with the chatbot?", value=st.session_state['expectation'])


            if st.form_submit_button("Submit"):
                st.session_state['gratitude_messages'].append({"role": "system", "content": f"User background: {st.session_state['background']}. Reason for visiting: {st.session_state['reason']}. User expectation: {st.session_state['expectation']}. I will call you Jamie, I'd like to start a gratitude exercise today. Can you help guide me through this process, asking me about what I'm grateful for and helping me explore why these things are meaningful to me?"})
                st.session_state['context_provided'] = True
                st.experimental_rerun()

        if st.button('Go to Home'):
            st.session_state['selected_app'] = None
            st.experimental_rerun()

    if st.session_state['context_provided']:
        
        st.write('You can now begin your conversation with Jamie! Feel free to send your first message.')

        # Container for chat history
        response_container = st.container()
        # Container for text box
        container = st.container()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_area("You:", key='input', height=100)

                if st.form_submit_button("Send"):
                    output = generate_response(user_input)
                    st.session_state['gratitude_past'].append(user_input)
                    st.session_state['gratitude_generated'].append(output)
                    st.session_state['gratitude_model_name'].append("GPT-3.5")

            st.write('To initiate a fresh session with all records and conversation deleted, simply press the designated button.')

            if st.button("New Session"):
                st.session_state['gratitude_generated'] = []
                st.session_state['gratitude_past'] = []
                st.session_state['gratitude_messages'] = []
                st.session_state['gratitude_model_name'] = []
                st.session_state['context_provided'] = []
                st.experimental_rerun()

            if st.session_state['gratitude_generated']:
                with response_container:
                    for i in range(len(st.session_state['gratitude_generated'])): 
                        if i < len(st.session_state["gratitude_past"]):
                            message(st.session_state["gratitude_past"][i], is_user=True, key=str(i) + '_user')
                        message(st.session_state["gratitude_generated"][i], key=str(i))

        if st.button('Go to Home'):
            st.session_state['selected_app'] = None
            st.experimental_rerun()

                    
############################################################################################################

def mental_app_page(st):
    # Initialise session state variables
    if 'mental_app_generated' not in st.session_state:
        st.session_state['mental_app_generated'] = []
    if 'mental_app_past' not in st.session_state:
        st.session_state['mental_app_past'] = []
    if 'mental_app_messages' not in st.session_state:
        st.session_state['mental_app_messages'] = []
    if 'mental_app_model_name' not in st.session_state:
        st.session_state['mental_app_model_name'] = []
    if 'mental_app_analysis' not in st.session_state:
        st.session_state['mental_app_analysis'] = ""  # Add a session variable to store analysis result
    if 'katie_context_provided' not in st.session_state:
        st.session_state['katie_context_provided'] = False
        st.session_state['katie_background'] = ''
        st.session_state['katie_expectation'] = ''
        st.session_state['katie_reason'] = ''

    # Generate a response
    def generate_response(prompt):
        st.session_state['mental_app_messages'].append({"role": "user", "content": prompt})

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=st.session_state['mental_app_messages'],
            max_tokens=3000,
            temperature=0.6
        )
        response = completion.choices[0].message.content.strip()
        st.session_state['mental_app_messages'].append({"role": "assistant", "content": response})

        return response

    # Analyze user messages
    def generate_analysis():
        user_messages = st.session_state['mental_app_past'][-5:]  # Get the last 5 user messages
        merged_user_messages = " ".join(user_messages)  # Merge the user messages into a single prompt

        st.session_state['mental_app_messages'] = [
            {"role": "user", "content": merged_user_messages},
            {"role": "assistant", "content": "Hi! Let me analyze the information you've provided."}
        ]

        st.session_state['mental_app_analysis'] = generate_response("")  # Update the analysis session variable

    st.markdown('<h1 style="color: #0984e3; font-weight: bold; text-align: center;">🌠 Katie the Catalyst 🌠</h1>', unsafe_allow_html=True)
    st.markdown('')


    if not st.session_state['katie_context_provided']:
        with st.form(key='context_form'):

            st.write('Before you begin, please fill out these forms to help Jamie understand you better and provide more meaningful support based on your current situation.')
            st.write('All the information you provide in the forms will be deleted after you finish your session or refresh the page.')

            st.session_state['katie_background'] = st.text_area("What is your background?", value=st.session_state['katie_background'])
            st.session_state['katie_reason'] = st.text_area("What brought you here?", value=st.session_state['katie_reason'])
            st.session_state['katie_expectation'] = st.text_area("What do you expect with your conversation with the chatbot?", value=st.session_state['katie_expectation'])

            if st.form_submit_button("Submit"):
                st.session_state['mental_app_messages'].append({"role": "system", "content": f"User background: {st.session_state['katie_background']}. Reason for visiting: {st.session_state['katie_reason']}. User expectation: {st.session_state['katie_expectation']}. Hi chatgpt, your name is Katie, I will call you Katie, I'd like you to act as an intelligent and empathetic psychotherapist for me today. Can you help guide me through a process of self-reflection, asking me questions that will help me better understand my own thoughts and feelings? I'd like you to gather as much information as possible about my mental state and current circumstances. Then, I'd like you to analyze this information and provide me with detailed insights, not a diagnosis, but a deeper understanding of my mental state. These insights should be expansive, precise, and supported by scientific evidence. I'd like our interaction to be conversational and comfortable."})
                st.session_state['katie_context_provided'] = True
                st.experimental_rerun()

        if st.button('Go to Home'):
            st.session_state['selected_app'] = None
            st.experimental_rerun()


    if st.session_state['katie_context_provided']:

        st.write('You can now begin your conversation with Katie! Feel free to send your first message.')

        # Container for chat history
        response_container = st.container()
        # Container for text box
        container = st.container()

        # New Session button
        if st.button("New Session"):
            st.session_state['mental_app_generated'] = []
            st.session_state['mental_app_past'] = []
            st.session_state['mental_app_messages'] = []
            st.session_state['mental_app_model_name'] = []
            st.session_state['mental_app_analysis'] = ""  # Reset the analysis session variable
            st.session_state['katie_context_provided'] = []
            st.session_state['katie_background'] = ''
            st.session_state['katie_expectation'] = ''
            st.session_state['katie_reason'] = ''
            st.experimental_rerun()

        with container:
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_area("You:", key='input', height=100)
                submit_button = st.form_submit_button(label='Send')

            if submit_button and user_input:
                output = generate_response(user_input)
                st.session_state['mental_app_past'].append(user_input)
                st.session_state['mental_app_generated'].append(output)
                st.session_state['mental_app_model_name'].append("GPT-3.5")


            st.write("When you are ready and have provided enough information to Katie, click the button below to generate an analysis.")

            # Replace the if condition for the analysis button with a permanent button
            analysis_button = st.button("Generate Analysis")
            if analysis_button:
                generate_analysis()
                st.experimental_rerun()

            st.write('To initiate a fresh session with all records and conversation deleted, simply press the designated button.')

        if st.session_state['mental_app_generated']:
            with response_container:
                for i in range(len(st.session_state['mental_app_generated'])):
                    if i < len(st.session_state["mental_app_past"]):
                        message(st.session_state["mental_app_past"][i], is_user=True, key=str(i) + '_user')
                    message(st.session_state["mental_app_generated"][i], key=str(i))

        # Add a new section for displaying the analysis
        if st.session_state['mental_app_analysis']:
            st.markdown(f"## Analysis")
            st.markdown(st.session_state['mental_app_analysis'])

        if st.button('Go to Home'):
            st.session_state['selected_app'] = None
            st.experimental_rerun()



############################################################################################################

def encourager_page(st):
    
    # Initialise session state variables
    if 'encourager_generated' not in st.session_state:
        st.session_state['encourager_generated'] = []
    if 'encourager_past' not in st.session_state:
        st.session_state['encourager_past'] = []
    if 'encourager_messages' not in st.session_state:
        st.session_state['encourager_messages'] = []
    if 'encourager_model_name' not in st.session_state:
        st.session_state['encourager_model_name'] = []

    if 'nathan_context_provided' not in st.session_state:
        st.session_state['nathan_context_provided'] = False
        st.session_state['encourager_background'] = ''
        st.session_state['encourager_reason'] = ''
        st.session_state['encourager_expectation'] = ''

    # Generate a response
    def generate_response(prompt):
        st.session_state['encourager_messages'].append({"role": "user", "content": prompt})

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=st.session_state['encourager_messages'],
            max_tokens=3000,
            temperature=0.6
        )
        response = completion.choices[0].message.content.strip()
        st.session_state['encourager_messages'].append({"role": "system", "content": response})

        return response

    st.markdown('<h1 style="color: #fdcb6e; font-weight: bold; text-align: center;">🌠 Nathan the Coach 🌠</h1>', unsafe_allow_html=True)
    st.markdown('')

    if not st.session_state['nathan_context_provided']:
        with st.form(key='context_form'):

            st.write('Before you begin, please fill out these forms to help Jamie understand you better and provide more meaningful support based on your current situation.')
            st.write('All the information you provide in the forms will be deleted after you finish your session or refresh the page.')

            st.session_state['encourager_background'] = st.text_area("What is your background?", value=st.session_state['encourager_background'])
            st.session_state['encourager_reason'] = st.text_area("What brought you here?", value=st.session_state['encourager_reason'])
            st.session_state['encourager_expectation'] = st.text_area("What do you expect with your conversation with the chatbot?", value=st.session_state['encourager_expectation'])

            if st.form_submit_button("Submit"):
                st.session_state['encourager_messages'].append({"role": "system", "content": f"User background: {st.session_state['encourager_background']}. Reason for visiting: {st.session_state['encourager_reason']}. User expectation: {st.session_state['encourager_expectation']}. Hello chatgpt, I will call you nathan, I'd like you to act as a supportive life coach for me today. Can you help guide me through a process of self-encouragement, asking me questions that will help me focus on the positive aspects of my life and believe in myself?"})
                st.session_state['nathan_context_provided'] = True
                st.experimental_rerun()
        if st.button('Go to Home'):
            st.session_state['selected_app'] = None
            st.experimental_rerun()



    if st.session_state['nathan_context_provided']:
        # Container for chat history
        response_container = st.container()
        # Container for text box
        container = st.container()
        with container:

            st.write('You can now begin your conversation with Nathan! Feel free to send your first message.')
            
            with st.form(key='my_form', clear_on_submit=True):
                user_input = st.text_area("You:", key='input', height=100)

                if st.form_submit_button("Send"):
                    output = generate_response(user_input)
                    st.session_state['encourager_past'].append(user_input)
                    st.session_state['encourager_generated'].append(output)
                    st.session_state['encourager_model_name'].append("GPT-3.5")

            st.write('To initiate a fresh session with all records and conversation deleted, simply press the designated button.')

            if st.button("New Session"):
                st.session_state['encourager_generated'] = []
                st.session_state['encourager_past'] = []
                st.session_state['encourager_messages'] = []
                st.session_state['encourager_model_name'] = []
                st.session_state['nathan_context_provided'] = False
                st.experimental_rerun()

            if st.session_state['encourager_generated']:
                with response_container:
                    for i in range(len(st.session_state['encourager_generated'])): 
                        if i < len(st.session_state["encourager_past"]):
                            message(st.session_state["encourager_past"][i], is_user=True, key=str(i) + '_user')
                        message(st.session_state["encourager_generated"][i], key=str(i))

        if st.button('Go to Home'):
            st.session_state['selected_app'] = None
            st.experimental_rerun()

############################################################################################################

# Create a dictionary to map app names to their corresponding functions
apps = {
    "Gratitude": gratitude_page,
    "Analyzer": mental_app_page,
    "Encourager": encourager_page
}

# Initialize session_state if it doesn't exist
if 'selected_app' not in st.session_state:
    st.session_state['selected_app'] = None

if st.session_state['selected_app'] is None:

    st.image('logo.png')
    st.write('**Introducing the resiliAInce, your AI powered mental health chatbot! Within this platform, you\'ll have the opportunity to chat with three chatbots: Jamie, Katie, and Nathan. These chatbots, powered by language models, are designed to support you in various aspects of your mental health.**')

    st.write('**This platform serves as a demonstration of how AI can complement mental health support, providing a starting point for those seeking assistance. It is highly emphasized that this is NOT A SUBSTITUTE to mental health support from professionals.**')

    st.markdown("Developed by [Eys](https://www.linkedin.com/in/acecanacan/)", unsafe_allow_html=True)

    st.markdown('<h1 style="color: #fd79a8; font-weight: bold; text-align: center;">🌱 Jamie the Grounded 🌱</h1>', unsafe_allow_html=True)
    st.write('Jamie the Grounded is your daily reminder of the good things in life. As a dedicated gratitude companion, she pushes you to engage in daily gratitude exercises to stay grounded and appreciative of what you have.')
    st.write('Her goal is to help you realize the positive aspects of your life consistently. With her, gratitude is not an occasional activity but a daily habit. Her ultimate aim is to help you cultivate a sense of appreciation that you can carry throughout your day')

    if st.button('Chat with Jamie'):
        st.session_state['selected_app'] = "Gratitude"
        st.experimental_rerun()

    st.markdown('<h1 style="color: #0984e3; font-weight: bold; text-align: center;">🌠 Katie the Catalyst 🌠</h1>', unsafe_allow_html=True)
    st.write('Katie the Catalyst is your supportive life coach with a knack for in-depth psychotherapeutic analysis. She drives your journey of self-discovery, focusing on the technical side of mental health.')
    st.write('Katie’s primary function is to thoroughly analyze and interpret your thoughts and feelings, revealing insights about your personality and emotional state. She serves as your guide, helping you comprehend what you are experiencing, and provides direction for facing your challenges')

    if st.button('Chat with Katie'):
        st.session_state['selected_app'] = "Analyzer"
        st.experimental_rerun()

    st.markdown('<h1 style="color: #fdcb6e; font-weight: bold; text-align: center;">🥇 Nathan the Coach 🥇</h1>', unsafe_allow_html=True)
    st.write('Nathan the Coach is your personal motivator, always there to bolster your self-belief. He champions your unique qualities, helping you recognize and embrace your individual potential. The primary goal of Nathan is to help you see what makes you distinct and amazing.')
    st.write('With him, every interaction is a celebration of your uniqueness, a step towards realizing your full potential. As an encourager, his primary function is to uplift, spotlight your strengths, and cheer you on in your journey of self-growth.')

    if st.button('Chat with Nathan'):
        st.session_state['selected_app'] = "Encourager"
        st.experimental_rerun()

    st.markdown('<h1 style="text-align: center;">Data Privacy</h1>', unsafe_allow_html=True)
    st.write('The data inputted by users in this application is not collected by the developer. Regarding data privacy rules for the chatbot, they adhere to the terms and regulations of OpenAI.')

elif st.session_state['selected_app'] == "Gratitude":
    gratitude_page(st)
elif st.session_state['selected_app'] == "Analyzer":
    mental_app_page(st)
elif st.session_state['selected_app'] == "Encourager":
    encourager_page(st)
