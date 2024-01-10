import openai
import PyPDF2
import pdfplumber
from langchain.document_loaders import PyPDFLoader

openai.api_key = "API_KEY"

def extract_text_from_pdf(pdf_path):
    pdf_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            pdf_text += page.extract_text()
    return pdf_text

def ask_question(question, context):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=f"Read the following text and answer the question:\n{context}\n\nQuestion: {question}\nAnswer:",
        max_tokens=50,
        temperature=0.2,
        stop=None
    )
    answer = response.choices[0].text.strip()
    return answer


loader=PyPDFLoader("Meta-12.31.2022-Exhibit-99.1-FINAL.pdf")
pdf_text = loader.load_and_split()[0:5]


while(True):
    question = input('Enter a prompt or type "quit": ')
    if (question == 'quit'):
        break;
    answer = ask_question(question, pdf_text)
    print("Answer:", answer)




