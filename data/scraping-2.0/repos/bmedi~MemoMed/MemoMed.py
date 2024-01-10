# a mock conversation dialogue for a clinical visit:

# Patient: Doctor, I've been feeling really tired lately, more than usual. I just don't have the energy to do anything.
# Doctor: I see. Have you noticed any other symptoms?
# Patient: Yes, I've also been experiencing shortness of breath, even when I'm not doing anything strenuous. And I've noticed that my heart has been beaing faster than normal.
# Doctor: "How about your appetite? Any changes there?
# Patient: Now that you mention it, I haven't been eating as much as I usually do. I've also been feeling a bit dizzy and lightheaded at times.

# Example questions:

# what are the major symptoms? 
# what might cause these symptoms?
# which specialist I should see? 


import openai
import speech_recognition as sr
import streamlit as st
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

def main():
    print(f" ")

if __name__ == "__main__":
    main()

# Initialize the conversation history
conversation_history = ChatMessageHistory()

def transcribe_speech():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        print("Speak Anything :")
        audio = r.listen(source)

        try:
            text = r.recognize_google(audio)
            print("You said : {}".format(text))
            return text
        except:
            print("Sorry could not recognize your voice")
            return None

def transcribe_audio(audio_file):
    r = sr.Recognizer()

    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)

    try:
        text = r.recognize_google(audio)
        st.write("Transcript: ", text)
        return text
    except:
        st.write("Sorry, I could not transcribe the file.")
        return None

# def generate_notes(transcribed_text):
#     template = f"""
#     <b>Patient:</b> {transcribed_text}

#     <b>Source of Information:</b>
#     <b>Date and Time:</b>
#     <b>Interpreter/Substitute Decision-Maker:</b>

#     <b>Allergies:</b>

#     <b>Relevant History and Physical Findings:</b>

#     <b>Vital Signs:</b>

#     <b>Pertinent Positive/Negative Findings:</b>

#     <b>Assessment of Patient Capacity:</b>

#     <b>Clinical Assessment:</b>

#     <b>Working Diagnosis:</b>
#     <b>Differential Diagnosis:</b>
#     <b>Final Diagnosis:</b>

#     <b>Plan of Action:</b>

#     <b>Investigations:</b>
#     <b>Consultations:</b>
#     <b>Treatment:</b>
#     <b>Follow-Up:</b>

#     <b>Rationale for the Plan:</b>

#     <b>Expectations of Outcomes:</b>

#     <b>Medications (Doses and Duration):</b>

#     <b>Medication Reconciliation:</b>

#     <b>Calls to Consultants:</b>
#     <b>Consultant's Name:</b>
#     <b>Advice Received:</b>

#     <b>Information Given by/to the Patient (or SDM):</b>

#     <b>Concerns Raised, Questions Asked, and Responses Given:</b>

#     <b>Verification of Patient Understanding:</b>

#     <b>Consent Discussion Summary:</b>

#     <b>Discharge Instructions:</b>

#     <b>Symptoms and Signs that Should Prompt a Reassessment:</b>

#     <b>Urgency of Follow Up:</b>

#     <b>Where and When to Return:</b>

#     <b>Missed Appointments:</b>

#     <b>Efforts to Follow Up on Investigation Results:</b>

#     <b>Communication with Other Care Providers at Discharge:</b>

#     <b>Signature of Writer and Role:</b>

#     -End of Note-

#     {{
#     - The generated note should not include any personally identifiable information (PII) or protected health information (PHI) that could violate privacy laws like HIPAA.
#     - The note should be factual and based on the information provided in the transcribed text.
#     - The note should not include any speculative or hypothetical information.
#     }}
#     """
#     # Use the OpenAI API to generate the note
#     response = openai.ChatCompletion.create(
#         model="gpt-4",
#         messages=[
#             {"role": "system", "content": template},
#             {"role": "user", "content": transcribed_text}
#         ]
#     )
#     return response['choices'][0]['message']['content'].strip()

def generate_notes(transcribed_text, template):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": template},
            {"role": "user", "content": transcribed_text}
        ]
    )
    suggestions = response['choices'][0]['message']['content'].strip()
    return suggestions


def generate_suggestions(note):
    messages = [
        {
            "role": "system",
            "content": "You are a helpful healthcare assistant that generates suggestions based on medical notes."
        },
        {
            "role": "user",
            "content": note
        }
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    suggestion = response['choices'][0]['message']['content']
    return suggestion

def chat_with_gpt(prompt, generated_notes, conversation_history):
    # Add the generated notes to the conversation history
    conversation_history.add_ai_message(generated_notes)

    # Add the user's message to the conversation history
    conversation_history.add_user_message(prompt)

    # Initialize the memory
    memory = ConversationBufferMemory()

    # Add the conversation history to the memory
    memory.chat_memory = conversation_history

    # Load the memory variables
    memory_variables = memory.load_memory_variables({})

    # Use the OpenAI API to generate the response
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": memory_variables['history']}
        ]
    )

    # Add the AI's message to the conversation history
    conversation_history.add_ai_message(response['choices'][0]['message']['content'])

    return response['choices'][0]['message']['content']


# Define the templates
templates = {
    "General": """
Patient: {transcribed_text}

Date and Time:
Reason for Visit:

Presenting Symptoms:

Past Medical History:

Physical Examination Findings:

Assessment:
Plan:

Signature:
""",
    "Pediatrician": """
Patient: {transcribed_text}

Date and Time of Visit:
Reason for Visit:

Presenting Symptoms:
Duration of Symptoms:

Past Medical History:
Immunization Status:
Growth and Development History:

Physical Examination Findings:
Vital Signs:
Growth Parameters:
Systemic Examination Findings:

Assessment:
Working Diagnosis:
Differential Diagnosis:

Plan:
Investigations:
Treatment Plan:
Follow-Up Plan:
Health Promotion and Disease Prevention Advice:

Parent's Concerns and Questions:
Responses Given:

Signature of Pediatrician:
""",
    "ED nurse": """
Patient: {transcribed_text}

Date and Time of Arrival:
Chief Complaint:

Presenting Symptoms:
Duration of Symptoms:

Past Medical History:
Allergies:

Vital Signs on Arrival:
Physical Examination Findings:

Nursing Assessment:
Level of Consciousness:
Pain Assessment:
Other Relevant Findings:

Interventions and Treatments Provided:
Medications Administered:
Procedures Performed:

Response to Interventions:
Change in Condition:

Handover Notes:

Signature of Nurse:
""",
    "Surgeon": """
Patient: {transcribed_text}

Date and Time of Consultation:
Reason for Consultation:

Presenting Symptoms:
Duration of Symptoms:

Past Medical History:
Previous Surgeries:
Allergies:

Physical Examination Findings:
Systemic Examination:
Local Examination:

Preoperative Diagnosis:
Differential Diagnosis:

Plan:
Investigations:
Proposed Surgical Procedure:
Risks and Benefits Discussed:

Consent Discussion Summary:

Postoperative Care Plan:

Patient's Concerns and Questions:
Responses Given:

Signature of Surgeon:
"""
}

def main():
    st.title("MemoMed: An Auto Note-Taking Tool for Doctors and Nurses")

    persona = st.selectbox("Select your persona:", ["General", "Pediatrician", "ED nurse", "Surgeon"])

    template = templates[persona]

    st.header("Transcribe Audio")

    # audio_file = st.file_uploader("Upload Audio", key='audio_file')
    # if st.button("Start Transcription"):
    #     st.button("Transcribing...", disabled=True)
    #     transcribed_text = transcribe_audio(audio_file)
    #     st.session_state.transcribed_text = transcribed_text
    #     st.write(transcribed_text)
    #     st.button("Transcription Complete", disabled=True)

    # Option for upload audio or using microphone
    option = st.selectbox("Choose an option", ["Upload Audio File", "Use Microphone"])

    if option == "Upload Audio File":
        audio_file = st.file_uploader("Upload Audio", type=['wav', 'mp3', 'flac'], key='audio_file')
        if st.button("Start Transcription from File"):
            st.button("Transcribing...", disabled=True)
            transcribed_text = transcribe_audio(audio_file)
            st.session_state.transcribed_text = transcribed_text
            st.write(transcribed_text)
            st.button("Transcription Complete", disabled=True)

    elif option == "Use Microphone":
        if st.button("Start Transcription from Microphone"):
            st.button("Transcribing...", disabled=True)
            transcribed_text = transcribe_speech()
            st.session_state.transcribed_text = transcribed_text
            st.write(transcribed_text)
            st.button("Transcription Complete", disabled=True)

    # Generate Notes
    st.header("Start of the Generate Notes")
    notes_input = st.text_area("Input", value=st.session_state.transcribed_text if 'transcribed_text' in st.session_state else '', key='notes_input')
    if st.button("Generate Notes"):
        notes = generate_notes(notes_input, template)
        st.session_state.notes = notes
        st.markdown(f"**{notes}**", unsafe_allow_html=True)

    # Generate Suggestions
    st.header("Start of the Generate Suggestions")
    suggestions_input = st.text_area("Input", value=st.session_state.notes if 'notes' in st.session_state else '', key='suggestions_input')
    if st.button("Generate Suggestions"):
        suggestions = generate_suggestions(suggestions_input)
        st.write(suggestions)

    st.title("MemoMed Chatbot")

    # User input
    user_input = st.text_input("Enter your message:")

    # Send button
    if st.button("Send"):
        response = chat_with_gpt(user_input, st.session_state.notes, conversation_history)
        st.write(f"AI: {response}")

if __name__ == "__main__":
    main()
