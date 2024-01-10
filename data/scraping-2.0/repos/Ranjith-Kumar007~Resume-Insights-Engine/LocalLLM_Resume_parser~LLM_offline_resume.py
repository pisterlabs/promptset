import re
from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from pypdf import PdfReader

reader = PdfReader("C:/Users/Ranjith/Desktop/Resumes/Ranjith_Resume.pdf")

# Print the number of pages in the PDF
print(f"There are {len(reader.pages)} Pages")

# Get the first page (index 0)
page = reader.pages[0]
# Use extract_text() to get the text of the page
text = page.extract_text()

# Make sure the model path is correct for your system!
model_path = "C:/Users/Ranjith/Downloads/gpt4all-falcon-q4_0.gguf"  # <-------- enter your model path here

template = """Text: {text}

Extracted Information:
Name: {name}
Phone Number: {phone}
Email: {email}
"""

prompt = PromptTemplate(template=template, input_variables=["text", "name", "phone", "email"])

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

n_gpu_layers = 40  # Change this value based on your model and your GPU VRAM pool.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of VRAM in your GPU.

llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    callback_manager=callback_manager,
    verbose=True,
)

llm_chain = LLMChain(prompt=prompt, llm=llm)

# Split the text into two chunks
chunk_size = len(text) // 2
chunks = [text[:chunk_size], text[chunk_size:]]

# Initialize variables to store extracted information
name, phone, email = "", "", ""

# Process each chunk
for chunk in chunks:
    # Pass the input as a dictionary
    input_data = {"text": chunk, "name": " ", "phone": " ", "email": " "}
    response = llm_chain.run(input_data)

    # Extract information using regular expressions
    name_match = re.search(r"Name: (.+)", response)
    phone_match = re.search(r"Phone Number: (.+)", response)
    email_match = re.search(r"Email: (.+)", response)

    # Update extracted information
    if name_match:
        name = name_match.group(1)
    if phone_match:
        phone = phone_match.group(1)
    if email_match:
        email = email_match.group(1)

# Print the final extracted information
print(f"Name: {name}\nPhone Number: {phone}\nEmail: {email}")
