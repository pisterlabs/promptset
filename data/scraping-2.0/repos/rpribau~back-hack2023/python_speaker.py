import zmq
import openai
import fitz
import os

# Load PDF file
pdf_file_path = 'testFile.pdf'  # Cambia 'example.pdf' al nombre de tu archivo PDF

# Verify if the PDF file exists
if not os.path.isfile(pdf_file_path):
    print(f"El archivo PDF '{pdf_file_path}' no existe.")
    exit(1)

# Open the PDF file and extract its text content
pdf_document = fitz.open(pdf_file_path)
pdf_text_content = ""
for page_num in range(pdf_document.page_count):
    page = pdf_document.load_page(page_num)
    pdf_text_content += page.get_text()

# Create a context object
context = zmq.Context()

# Import OpenAI credentials
openai.api_type = "azure"
openai.api_key = "6b25369971534252bbcee5e488ce59f1"
openai.api_base = "https://openaistkinno.openai.azure.com/"
openai.api_version = "2023-07-01-preview"

# Create a replier socket
sock = context.socket(zmq.REP)

# Bind to the JavaScript client
sock.bind("tcp://127.0.0.1:3000")

while True:
    # Receive a request message
    prompt = sock.recv()

    # Print the message
    print("Received: %s" % prompt.decode())

    msg = prompt.decode()

    # Include the PDF text content in the message to OpenAI
    messages = [
        {
            "role": "system",
            "content": "This is the PDF content:\n" + pdf_text_content
        },
        {
            "role": "user",
            "content": msg,
        },
    ]

    # Create the completion using the combined messages
    response = openai.ChatCompletion.create(
        engine="InnovationGPT2",
        messages=messages,
        max_tokens=1000
    )

    # Print the output
    print(response.choices[0].message.content)

    # Send a reply message
    sock.send_string(response.choices[0].message.content)
