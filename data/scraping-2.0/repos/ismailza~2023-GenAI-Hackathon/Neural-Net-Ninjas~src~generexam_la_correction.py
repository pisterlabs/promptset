import streamlit as st
import pdfplumber
import spacy
import openai

# Load the spaCy model
try:
    nlp = spacy.load("fr_core_news_sm")
except OSError:
    # If the model is not found, download it
    print("Model 'fr_core_news_sm' not found. Downloading...")
    spacy.cli.download("fr_core_news_sm")
    nlp = spacy.load("fr_core_news_sm")

# Set your OpenAI API key
openai.api_key = "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"

# Function to extract information from PDF
def extract_information_from_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        pdf_text = ""
        for page in pdf.pages:
            pdf_text += page.extract_text()
    return pdf_text

# Function to generate questions using GPT-3
def generate_questions_using_gpt3(text):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"PDF content: {text}\nGenerate questions:",
        temperature=0.5,
        max_tokens=200
    )
    return response.choices[0].text.strip()

# Function to query GPT-3 for responses
def query_gpt3(question):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Question: {question}\nAnswer:",
        temperature=0.5,
        max_tokens=200
    )
    return response.choices[0].text.strip()

# Function to correct user answers
def correct_user_answers(generated_questions, user_answers):
    corrections = {}
    for question in generated_questions.split('\n'):
        if question:
            # Assuming the question is followed by a line break, remove it
            question = question.replace('\n', '')
            user_answer = user_answers.get(question, "")
            correction = query_gpt3(user_answer)
            corrections[question] = correction
    return corrections

# Streamlit app
def main():
    st.title("Générateur de Questions et Correction de Réponses")

    uploaded_file = st.file_uploader("Téléchargez un fichier PDF", type=["pdf"])

    if uploaded_file is not None:
        # Extract information from the PDF
        pdf_text = extract_information_from_pdf(uploaded_file)

        # Generate questions based on the PDF content
        generated_questions = generate_questions_using_gpt3(pdf_text)

        st.subheader("Questions générées:")
        st.write(generated_questions)

        # Allow the user to enter answers
        user_answers = {}
        for question in generated_questions.split('\n'):
            if question:
                # Assuming the question is followed by a line break, remove it
                question = question.replace('\n', '')
                user_answer = st.text_input(f"Réponse pour la question: {question}")
                user_answers[question] = user_answer

        # Correct user answers
        corrections = correct_user_answers(generated_questions, user_answers)

        st.subheader("Corrections des Réponses:")
        for question, correction in corrections.items():
            st.write(f"Question: {question}")
            st.write(f"Réponse de l'utilisateur: {user_answers[question]}")
            st.write(f"Correction : {correction}")
            st.write("---")

if __name__ == "__main__":
    main()
