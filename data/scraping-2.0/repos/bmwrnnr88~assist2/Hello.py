import openai
import streamlit as st
from bs4 import BeautifulSoup
import requests
import pdfkit
import time

# Initialize the OpenAI client with API key and Assistant ID from secrets.toml
openai.api_key = st.secrets["OPENAI_API_KEY"]
assistant_id = st.secrets["ASSISTANT_ID"]

client = openai

def main():

    if 'client' not in st.session_state:
        # Initialize the client
        st.session_state.client = openai.OpenAI()

        st.session_state.file = st.session_state.client.files.create(
            file=open("songs.txt", "rb"),
            purpose='assistants'
        )

        # Step 1: Create an Assistant
        st.session_state.assistant = st.session_state.client.beta.assistants.create(
            name="Customer Service Assistant",
            instructions="""You are the Sarcastic Vocab Wizard who assess the user on their knowledge of the assigned vocabulary words below. The Sarcastic Vocab Wizard is designed to combine a playful, mildly mocking tone with a trial-and-error approach to vocabulary learning. At the beginning of the quiz, the wizard will present a specific vocabulary word from the weekly list. The student is then asked to use this word in a sentence. The sentence must demonstrate knowledge of the word, meaning the sentence must be more than grammatically correct. The correct sentence must also have enough information that it demonstrates understanding of the word. If the sentence is not quite right, the wizard will provide sarcastic yet constructive feedback, encouraging the student to try again. The wizard allows multiple attempts before revealing an example, fostering independent learning. After going through all the words, the wizard will revisit any words that required revealing an example for another try. This approach ensures that humor is used to enhance the learning experience, while also making sure that students truly understand the words they are using., 
            The assigned  vocabulary words are: Abate: (verb) to become less active, less intense, or less in amount. Example sentence: As I began my speech, my feelings of nervousness quickly abated​.,
            Abstract: (adjective) existing purely in the mind; not representing actual reality. Example sentence: Julie had trouble understanding the appeal of the abstract painting​.,
            Abysmal: (adjective) extremely bad. Example sentence: I got an abysmal grade on my research paper​ which ruined my summer vacation.,
            Accordingly: (adverb) in accordance with. Example sentence: All students must behave accordingly, otherwise, they will receive harsh punishments​.,
            Acquisition: (noun) the act of gaining a skill or possession of something. Example sentence: Language acquisition is easier for kids than it is for adults.,
            Adapt: (verb) to make suit a new purpose; to accommodate oneself to a new condition, setting, or situation​. Begin all new conversations by greeting the user and then starting the vocab assessment.,
            Adept: (adjective) having knowledge or skill (usually in a particular area). Example sentence: Beth loves playing the piano, but she’s especially adept at the violin​.,
            Adequate: (adjective) having sufficient qualifications to meet a specific task or purpose. Example sentence: Though his resume was adequate, the company doubted whether he’d be a good fit​.,
            Advent: (noun) the arrival or creation of something (usually historic). Example sentence: The world has never been the same since the advent of the light bulb​.,
            Adversarial: (adjective) relating to hostile opposition. Example sentence: An adversarial attitude will make you many enemies in life​.""",
            model="gpt-4-1106-preview",
            file_ids=[st.session_state.file.id],
            tools=[{"type": "retrieval"}]
        )

        # Step 2: Create a Thread
        st.session_state.thread = st.session_state.client.beta.threads.create()

    user_query = st.text_input("Enter your query:", "Welcome! What is your name, first and last, and the period you have the wonderful Mr. Ward!")

    if st.button('Submit'):
        # Step 3: Add a Message to a Thread
        message = st.session_state.client.beta.threads.messages.create(
            thread_id=st.session_state.thread.id,
            role="user",
            content=user_query
        )

        # Step 4: Run the Assistant
        run = st.session_state.client.beta.threads.runs.create(
            thread_id=st.session_state.thread.id,
            assistant_id=st.session_state.assistant.id,
            instructions="Please address the user as Mr. Ward's Favorite Student"
        )

        while True:
            # Wait for 5 seconds
            time.sleep(5)

            # Retrieve the run status
            run_status = st.session_state.client.beta.threads.runs.retrieve(
                thread_id=st.session_state.thread.id,
                run_id=run.id
            )

            # If run is completed, get messages
            if run_status.status == 'completed':
                messages = st.session_state.client.beta.threads.messages.list(
                    thread_id=st.session_state.thread.id
                )

                # Loop through messages and print content based on role
                for msg in messages.data:
                    role = msg.role
                    content = msg.content[0].text.value
                    st.write(f"{role.capitalize()}: {content}")
                break
            #else:
                #st.write("Waiting for the Assistant to process...")
                #time.sleep(5)

if __name__ == "__main__":
    main()
