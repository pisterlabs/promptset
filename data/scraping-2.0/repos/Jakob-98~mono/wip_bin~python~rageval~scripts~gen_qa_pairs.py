from langchain.text_splitter import CharacterTextSplitter
import os
import PyPDF2
import openai

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=3000, chunk_overlap=400, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def read_pdf(file_path):
    pdf_text = ""

    with open(file_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        number_of_pages = len(pdf_reader.pages)

        for page_num in range(number_of_pages):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            pdf_text += page_text

    return pdf_text

def pdfs_from_folder(folder_path):
    pdf_texts = []  # List to store the text content of each PDF

    for filename in os.listdir(folder_path):
        if filename.endswith('.pdf'):
            if not "Ford" in filename:
                continue
            file_path = os.path.join(folder_path, filename)
            pdf_text = read_pdf(file_path)
            pdf_texts.append(pdf_text)

    return pdf_texts


SYSTEM_PROMPT = """
You are an AI assistant part of a system designed to generate question-answer pairs for domain specific documents. The purpose is to extract a factual question and answer relevant to the information in a given document. You should also rate the relevance of the question on a scale of 0 to 1. If the given document has no factual information, generate a question with a relevance of 0. Your answer must adhere to the JSON structure provided in the example. 

Example INPUT: "The Eiffel Tower, located in Paris, France, is one of the most iconic landmarks in the world. Designed by the French engineer Gustave Eiffel and completed in 1889. The Eiffel Tower has three levels, with restaurants on the first and second levels and an observation deck at the top. The tower is 330 meters (1,083 feet) tall, including its antennas, and was the tallest man-made structure in the world when it was completed."

Example OUTPUT: "{
    "question": "what is the length of the eiffel tower?"
    "answer": "the length of the eiffel tower is 330 meters"
    "relevance": 1
}"
"""
openai.api_type = "azure"
openai.api_key = ...
openai.api_base = "https://aml-testopenai-jakob-aoai.openai.azure.com/"
openai.api_version = "2023-07-01-preview"

def chat_complete(messages):
    return openai.ChatCompletion.create(
  engine="gpt35deployment",
  messages = messages,
  temperature=0.7,
  max_tokens=800,
  top_p=0.95,
  frequency_penalty=0,
  presence_penalty=0,
  stop=None)

get_messages = lambda m: [
	{
		"role": "system",
		"content": SYSTEM_PROMPT
	},
    {
        "role": "user",
        "content": f"INPUT: {m}\n OUTPUT:"
    }
    ]

if __name__ == "__main__":
    folder_path = "./data/pdf"
    all_pdf_texts = pdfs_from_folder(folder_path)
    for chunk in get_text_chunks(all_pdf_texts[0])[0:10]:
        print(chat_complete(get_messages(chunk)))
