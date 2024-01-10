import streamlit as st
import openai
import tempfile
import librosa
import subprocess
import io
from faster_whisper import WhisperModel
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# Access the API key from the environment variable
api_key = os.getenv("OPENAI_API_KEY")

# Set the API key for the OpenAI library
openai.api_key = api_key

# Title
st.title("Audio Transcribing and Analysis")

# Upload widget
uploaded_file = st.file_uploader("Upload your audio file for analysis", type=["wav", "mp3", "m4a"])

# Buttons
process_button = st.button("Transcribe")

model_size = "large-v2"

# Run on CPU with FP16
model = WhisperModel(model_size, device="cpu", compute_type="int8")

# Session state to store responses
if not hasattr(st.session_state, 'transcription'):
    st.session_state.transcription = None
if not hasattr(st.session_state, 'interview_conversation'):
    st.session_state.interview_conversation = None
if not hasattr(st.session_state, 'insights'):
    st.session_state.insights = None
if not hasattr(st.session_state, 'rating'):
    st.session_state.rating = None
if not hasattr(st.session_state, 'responses'):
    st.session_state.responses = []
if not hasattr(st.session_state, 'form_data'):
  st.session_state.form_data = {}


def convert_to_wav(audio_file):
    # Write audio bytes to temp file
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(audio_file.getvalue())

    # Convert using ffmpeg
    subprocess.call(f"ffmpeg -i {f.name} temp.wav", shell=True)

    return "temp.wav"

def generate_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt +'\n\n###\n\n',
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0,
    )
    return response.choices[0].text


if process_button:
    if uploaded_file:
        audio_file = uploaded_file.read()

        # Convert to WAV
        wav_file = convert_to_wav(io.BytesIO(audio_file))

        # Load converted WAV
        audio, sr = librosa.load(wav_file)

        # Display audio characteristics
        audio_format = uploaded_file.type
        audio_length = librosa.get_duration(y=audio, sr=sr)
        audio_size = len(audio_file) / 1024  # Convert to KB

        st.subheader("Audio Characteristics:")
        st.write(f"Format: {audio_format}")
        st.write(f"Length: {audio_length:.2f} seconds")
        st.write(f"Size: {audio_size:.2f} KB")

        # Display play button
        st.audio(audio_file)

        # Notify that audio processing has started
        st.text("Processing audio...")

        # Transcribe
        transcription_result, _ = model.transcribe(audio)
        transcription_text = ' '.join([segment.text for segment in transcription_result])

        # Display transcribed text
        #st.subheader("Transcribed text:")
        #st.write(transcription_text)

        # Create a prompt for the conversation
        prompt = f"""
        Transcribed call recording conversation: {transcription_text}
        Context: This is a conversation between two speakers where a company executive from Ketto is interviewing a user for the monthly donation product by Ketto.
        Instruction: Convert this entire text into a conversational format between two people.
        Example: Speaker 1: Starts the conversation (Company executive)
                 Speaker 2: Gives a response to speaker 1 (Customer)
        """

        # Generate the interview conversation using the new function
        interview_conversation = generate_response(prompt)



        # Display generated conversation
        st.session_state.interview_conversation = interview_conversation
        st.subheader("Generated Interview Conversation:")
        st.write(st.session_state.interview_conversation )

        # Important insights about the conversation
        prompt_insights = f"""
        Transcribed call recording conversation: {interview_conversation}
        Context: This is a conversation between two speakers where a company executive from Ketto is interviewing a user for the monthly donation product by Ketto.
        Instruction: Give me 5 important insights such as what could be the problem faced by speaker 2 (customer) and how we could improve and make this experience better for the user. The challenges will be what things the company executive could have done better.
        Example: Challenges: 1. Timing and Health Status: Speaker 2 recently underwent a health check and expressed that they do not need the plan immediately. This suggests that the current health status and timing might not align with the benefits.
                 Solutions: 1. Customization
        """

        # Generate insights using the new function
        st.session_state.insights = generate_response(prompt_insights)

        # Display insights
        st.subheader("Improvements:")
        st.write(st.session_state.insights)

        # General rating on various parameters
        prompt_rating = f"""
        Transcribed call recording conversation: {interview_conversation}
        Context: This is a conversation between two speakers where a company executive from Ketto is interviewing a user for the monthly donation product by Ketto.
        Instruction: Rate the interviewer (1-10) based on these parameters with a simple explanation:
        1. Interviewer Rating
        2. Estimated Time for Conversation
        3. Customer Satisfaction
        4. Interviewer Politeness
        5. Interviewer's Ability to Answer Queries
        6. Overall Interviewer Performance
        Example: Interviewer Rating (8/10)
        The interviewer seems to be engaging and polite throughout the conversation.
        """

        # Generate rating using the new function
        st.session_state.rating = generate_response(prompt_rating)

        # Display rating
        st.subheader("Overall Analysis:")
        st.write(st.session_state.rating)

        # "Ask Us" section
        st.title("Ask Us")
        with st.form("ask_us_form"):
            user_question = st.text_input("Ask a question:")
            submit_button = st.form_submit_button("Submit")
            if submit_button and user_question:
                # Generate response using user's question and interview conversation
                user_question_prompt = f"User Question: {user_question}, conversation: {st.session_state.interview_conversation}"
                user_response = generate_response(user_question_prompt)
                st.subheader("Your Question:")
                st.write(user_question)
                st.subheader("Response:")
                st.write(user_response)
                st.session_state.responses.append((user_question, user_response))

        # Display user responses
        st.title("User Responses")
        for question, response in st.session_state.responses:
            st.subheader("Question:")
            st.write(question)
            st.subheader("Response:")
            st.write(response)
