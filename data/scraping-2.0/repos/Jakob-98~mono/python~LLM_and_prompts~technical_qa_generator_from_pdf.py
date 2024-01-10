from langchain.text_splitter import CharacterTextSplitter
import os
import PyPDF2
import openai
import json

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
            file_path = os.path.join(folder_path, filename)
            pdf_text = read_pdf(file_path)
            pdf_texts.append(pdf_text)

    return pdf_texts


SYSTEM_PROMPT = """
You are an AI whose purpose it is to generate question and answer pairs.

It is crucial these question answer pairs are specfic to the context the USER will give you and are related to TECHNICAL content, such that these question answer pairs cannot be retrieved otherwise. DO NOT make up questions and answers that are not related to the context the USER will give you, this will be heavily penalized.

If no technical question can be formulated, it is acceptable to return none. You are expected to return the question pair in JSON like so:

{
    "question": "What is the operating pressure of TK-3413?",
    "answer": "The operating pressure is 1.5 bar."
}

Examples:
USER:
"TK-3413 is a pressure vessel that is used to store water. It is used in the production of the Ford F-150. The operating pressure is 1.5 bar."
AI:
{
     "question": "What is the operating pressure of TK-3413?",
     "answer": "The operating pressure is 1.5 bar."
}
USER:
"The captial of France Paris, in Paris lays the Eiffel Tower. The Eiffel Tower is 324 meters tall."
AI:
{
     "question": "NONE", # No technical question can be formulated, and any search engine can retrieve this information, so None must be returned.
     "answer": "NONE."
}

"""
openai.api_type = "azure"
openai.api_key = "YOUR_KEY"
openai.api_base = "YOUR_ENDPOINT"
openai.api_version = "2023-07-01-preview"

def chat_complete(messages):
    return openai.ChatCompletion.create(
  engine="gpt4-32k-aoai-caneast",
  messages = messages,
  temperature=0.1,
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
        "content": f"USER: {m}"
    }
]

if __name__ == "__main__":
    folder_path = "./"
    all_pdf_texts = pdfs_from_folder(folder_path)
    qa_pairs = []
    for chunk in get_text_chunks(all_pdf_texts[0])[0:100]: #NOTE: notice the limit
        response = chat_complete(get_messages(chunk))
        try:
            response = json.loads(response['choices'][0]['message']["content"])
        except:
            continue
        qa_pairs.append(response)
    print(qa_pairs)