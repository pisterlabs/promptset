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

# Function to extract questions from text
def extract_questions(text):
    doc = nlp(text)
    questions = [sent.text.strip() for sent in doc.sents if sent.text.strip().endswith("?")]
    return questions

# Function to query GPT-3 for responses
def query_gpt3(question):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=f"Question: {question}\nAnswer:",
        temperature=0.5,
        max_tokens=200
    )
    return response.choices[0].text.strip()

# Function to extract and correct repeated questions using GPT-3
def extract_and_correct_questions(text):
    doc = nlp(text)
    questions = [sent.text.strip() for sent in doc.sents if sent.text.strip().endswith("?")]

    unique_questions = list(set(questions))
    corrected_questions = {}

    for question in unique_questions:
        correction = query_gpt3(question)
        corrected_questions[question] = correction

    return corrected_questions

# Streamlit app
def main():
    st.title("Analyse d'un PDF d'Examen")

    uploaded_file = st.file_uploader("Téléchargez un fichier PDF d'examen", type=["pdf"])

    if uploaded_file is not None:
        # Read PDF and extract text
        with pdfplumber.open(uploaded_file) as pdf:
            pdf_text = ""
            for page in pdf.pages:
                pdf_text += page.extract_text()

        # Extract and correct repeated questions from the text
        corrected_questions = extract_and_correct_questions(pdf_text)

        st.subheader("Questions les Plus Fréquentes et leurs Corrections:")
        for question, correction in corrected_questions.items():
            st.write(f"Question: {question}")
            st.write(f"Correction : {correction}")
            st.write("---")

if __name__ == "__main__":
    main()
