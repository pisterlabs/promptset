from fastapi import FastAPI, HTTPException
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
from pydantic import BaseModel
from typing import List, Optional
import requests
import os
import nltk
from io import BytesIO
from PyPDF2 import PdfReader
from transformers import GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM
import torch
from dotenv import load_dotenv
from reportlab.pdfgen import canvas
from fastapi import FastAPI, HTTPException, UploadFile, File
from transformers import GPT2Tokenizer
from pydantic import BaseModel
from typing import List, Optional
import requests
import os
import nltk
from io import BytesIO
# from PyPDF2 import PdfReader
from transformers import GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM
import torch
import torch.optim as optim


if torch.cuda.is_available(): 
    n_gpu = torch.cuda.device_count()
    print(f"You have {n_gpu} GPUs available.")

    for i in range(n_gpu):
        print(f"\nGPU {i}:")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(i)/1e6:.2f} MB")
        print(f"Memory Cached: {torch.cuda.memory_reserved(i)/1e6:.2f} MB")
else:
    print("No GPU available.")

import torch.nn as nn
from dotenv import load_dotenv
import openai
from tika import parser
import GPUtil
nltk.download('punkt')

IBM_API_KEY = os.getenv("IBM_API_KEY")  # read IBM API key from .env file

# Load the desired model and its tokenizer
# MODEL_NAME_GPT3 = "EleutherAI/gpt-neox-20b"
# MODEL_NAME_GPT2 = "gpt2"
# tokenizer_gpt3 = AutoTokenizer.from_pretrained(MODEL_NAME_GPT3)
# model_gpt3 = AutoModelForCausalLM.from_pretrained(MODEL_NAME_GPT3)
# tokenizer_gpt2 = GPT2Tokenizer.from_pretrained(MODEL_NAME_GPT2)
# model_gpt2 = GPT2LMHeadModel.from_pretrained(MODEL_NAME_GPT2)
app = FastAPI()

default_file_path = 'https://rfp-cases.s3.ca-tor.cloud-object-storage.appdomain.cloud/2023/Bell/C09AD188-0000-C922-B40E-E6AA3F3B9A1F/906090ae-344d-4108-a5d8-9ca42f3fc292/906090ae-344d-4108-a5d8-9ca42f3fc292.pdf'

load_dotenv()  # take environment variables from .env.
openai.api_key = os.getenv("OPENAI_API_KEY")

    # ###################################################OpenAssistant/llama2-13b-orca-8k-3319 pdf

class Item(BaseModel):
    text: str = None

class PDFData(BaseModel):
    pdf_url: str

def extract_text_from_pdf(file_as_bytes):
    pdf = PdfReader(file_as_bytes)
    text = ""
    for page in range(len(pdf.pages)):

        text += pdf.pages[page].extract_text()
    return text

def generate_token_chunks(tokens, chunk_size):
    return [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]

@app.post("/analyze_pdf_llama2_13b_orca_8k/")
async def analyze_pdf_llama2_13b_orca_8k(pdf_data: PDFData=r"{  'pdf_url': 'https://rfp-cases.s3.ca-tor.cloud-object-storage.appdomain.cloud/2023/Bell/C09AD188-0000-C922-B40E-E6AA3F3B9A1F/906090ae-344d-4108-a5d8-9ca42f3fc292/906090ae-344d-4108-a5d8-9ca42f3fc292.pdf'}"):
    response = requests.get(pdf_data.pdf_url)
    response.raise_for_status()

    file_as_bytes = BytesIO(response.content)
    text = extract_text_from_pdf(file_as_bytes)

    tokenizer = AutoTokenizer.from_pretrained("/root/.ssh/llama2-13b-orca-8k-3319", use_fast=False)
    model = AutoModelForCausalLM.from_pretrained("/root/.ssh/llama2-13b-orca-8k-3319", torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")

    tokens = tokenizer.encode(text)
    chunks = generate_token_chunks(tokens,2000)

    results = []
    for i, token_chunk in enumerate(chunks):
        chunk = tokenizer.decode(token_chunk)
        if i == len(chunks) - 1:  # run for the last chunk
            prompts = ["Could you help me extract the title from this document?", "Can you provide a summary of this document?", "Can you identify the due date mentioned in this document?", "What are the main requirements listed in this document?"]

            for prompt in prompts:
                inputs = tokenizer(f"{prompt}\n\n{chunk}", return_tensors="pt")
                inputs = {name: tensor.to("cuda") for name, tensor in inputs.items()}
                output = model.generate(**inputs, do_sample=True, top_p=0.95, top_k=0, max_new_tokens=256)
                output_text = tokenizer.decode(output[0], skip_special_tokens=True)
                results.append(output_text)
        else:  # for the first chunks, just instruct the model to read
            instruction = f"Please read this part of the document carefully:\n\n{chunk}"
            inputs = tokenizer(instruction, return_tensors="pt")
            inputs = {name: tensor.to("cuda") for name, tensor in inputs.items()}
            model.generate(**inputs, do_sample=True, top_p=0.95, top_k=0, max_new_tokens=256)  # we don't need the output here

    return {
        'title': results[0],
        'summary': results[1],
        'due_date': results[2],
        'requirements': results[3]
    }
    # ###################################################OpenAssistant/llama2-13b-orca-8k-3319 pdf

# ###################################################OpenAssistant/llama2-13b-orca-8k-3319 pdf
# from accelerate import Accelerator
# from transformers import AutoModelForCausalLM, AutoTokenizer

# accelerator = Accelerator()

# # Initialize the OpenAssistant model
# llama2_model_name = "OpenAssistant/llama2-13b-orca-8k-3319"
# tokenizer_llama2 = AutoTokenizer.from_pretrained("/root/.ssh/llama2-13b-orca-8k-3319")
# model = AutoModelForCausalLM.from_pretrained("/root/.ssh/llama2-13b-orca-8k-3319")
# optimizer = optim.Adam(model.parameters(), lr=0.001)

# # Using Accelerator for multi-GPU usage
# model, optimizer = accelerator.prepare(model, optimizer) # Assuming you have an optimizer defined somewhere

# try:
#     model.to(accelerator.device)
# except torch.cuda.OutOfMemoryError as e:
#     print(f"Error: {e}")

#     for i in range(accelerator.n_gpu):
#         print(f"\nGPU {i} after error:")
#         print(f"Memory Allocated: {torch.cuda.memory_allocated(i)/1e6:.2f} MB")
#         print(f"Memory Cached: {torch.cuda.memory_reserved(i)/1e6:.2f} MB")

# system_messages = ["As an AI assistant, read the RFP document carefully, understand its content, and provide a detailed, clear, and concise explanation of the title, summary, due date, and list of requirements."]

# def generate_text_llama2(prompt):
#     formatted_prompt = f"""{system_message}</s>{prompt}</s>"""
#     inputs = tokenizer_llama2(formatted_prompt, return_tensors="pt")
#     inputs = {name: tensor.to(accelerator.device) for name, tensor in inputs.items()}
    
#     # Monitor GPU usage before generating text
#     GPUs = GPUtil.getGPUs()
#     for i, gpu in enumerate(GPUs):
#         print(f'GPU {i}: {gpu.load*100}% | Memory Used: {gpu.memoryUsed:.2f}MB | Memory Total: {gpu.memoryTotal:.2f}MB')
        
#     # Use accelerator to perform backward operation
#     with accelerator.autocast():
#         outputs = model(**inputs)
#         loss = outputs.loss if isinstance(outputs, tuple) else outputs
#     accelerator.backward(loss)

#     output_text = tokenizer_llama2.decode(outputs[0], skip_special_tokens=True)
    
#     # Clear GPU cache after generating text
#     torch.cuda.empty_cache()

#     return output_text




# def extract_text_from_pdf(file_as_bytes):
#     pdf = PdfReader(file_as_bytes)
#     text = ""
#     for page in range(pdf.getNumPages()):
#         text += pdf.pages[page].extract_text()
#     return text

# def generate_token_chunks(tokens, chunk_size):
#     return [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]

# @app.get("/pdf/token_and_chunk_count")
# async def get_token_and_chunk_count(pdf_url: str = "default_file_path"):
#     response = requests.get(pdf_url)
#     response.raise_for_status()

#     file_as_bytes = BytesIO(response.content)
#     text = extract_text_from_pdf(file_as_bytes)

#     tokens = tokenizer_llama2.encode(text)
#     token_count = len(tokens)
#     chunks = generate_token_chunks(tokens,8000)
#     chunk_count = len(chunks)

#     return {"token_count": token_count, "chunk_count": chunk_count}

# @app.post("/pdf/analyze_llama2_8k")
# async def analyze_pdf_llama2_8k(pdf_url: str = default_file_path):
#     response = requests.get(pdf_url)
#     response.raise_for_status()

#     file_as_bytes = BytesIO(response.content)
#     rfp_text = extract_text_from_pdf(file_as_bytes)

#     with open('rfp_text_content.txt', 'w', encoding='utf-8') as f:
#         f.write(rfp_text)

#     tokens = tokenizer_llama2.encode(rfp_text)
#     token_count = len(tokens)
#     chunks = generate_token_chunks(tokens,4000)
#     chunk_count = len(chunks)

#     result = {}
#     for i, token_chunk in enumerate(chunks):
#         chunk = tokenizer_llama2.decode(token_chunk)
#         if i == len(chunks) - 1:
#             title_prompt = f"Could you help me extract the title from this RFP?\n\n{chunk}"
#             summary_prompt = f"Can you provide a summary of this RFP?\n\n{chunk}"
#             due_date_prompt = f"Can you identify the due date mentioned in this RFP?\n\n{chunk}"
#             requirements_prompt = f"What are the main requirements listed in this RFP?\n\n{chunk}"

#             title = generate_text_llama2(title_prompt)
 
#             for i in range(n_gpu):
#                 print(f"\nGPU {i} after loading data:")
#                 print(f"Memory Allocated: {torch.cuda.memory_allocated(i)/1e6:.2f} MB")
#                 print(f"Memory Cached: {torch.cuda.memory_reserved(i)/1e6:.2f} MB")

#             summary = generate_text_llama2(summary_prompt)
#             due_date = generate_text_llama2(due_date_prompt)
#             requirements = generate_text_llama2(requirements_prompt)

#             result = {
#                 'title': title,
#                 'summary': summary,
#                 'due_date': due_date,
#                 'requirements': requirements
#             }
#         else:
#             instruction = f"Please read this part of the RFP carefully:\n\n{chunk}"
#             generate_text_llama2(instruction)

#     return result
# ###################################################OpenAssistant/llama2-13b-orca-8k-3319 pdf
#####################################HuggingFace LLaMA-2-7B-32K simple
# llama2_model_name = "OpenAssistant/llama2-13b-orca-8k-3319"
# tokenizer_llama2 = AutoTokenizer.from_pretrained("/root/.ssh/llama2-13b-orca-8k-3319")
# model = AutoModelForCausalLM.from_pretrained("/root/.ssh/llama2-13b-orca-8k-3319")

class Item(BaseModel):
    text: str = None

@app.post("/generate_text_HuggingFace_llama2_13b_orca_8k/")
async def generate_text_HuggingFace_llama2_13b_orca_8k(item: Item):
    """
    {"text": "where is capital of Iran"}   
    """
    tokenizer = AutoTokenizer.from_pretrained("/root/.ssh/llama2-13b-orca-8k-3319", use_fast=False)
    model = AutoModelForCausalLM.from_pretrained("/root/.ssh/llama2-13b-orca-8k-3319", torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")

    system_message = "You are a helpful, respectful and honest assistant. please Read This RFP document and return title, summary, dates, and list of requirements"

    # If no input provided, read from the file
    if item.text is None:
        with open('/root/.ssh/fastapi/rfp_text_content_8k.txt', 'r') as file:
            user_prompt = file.read()
            user_prompt = {"text": user_prompt}  # Wrap the text in a dictionary similar to a JSON
    else:
        # Extract the text from the input JSON
        user_prompt = item.text

    # Format the input for the model
    prompt = f"""{system_message}</s>{user_prompt}</s>"""
    inputs = tokenizer(prompt, return_tensors="pt")

    # Make sure to send the tensors to the same device where your model is
    inputs = {name: tensor.to("cuda") for name, tensor in inputs.items()}

    # Generate a response
    output = model.generate(**inputs, do_sample=True, top_p=0.95, top_k=0, max_new_tokens=256)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Return the generated text
    return {"generated_text": output_text}

#####################################OpenAssistant/llama2-13b-orca-8k-3319 pdf
#####################################HuggingFace LLaMA-2-7B-32K simple


@app.post("/generate_text_HuggingFace_LLaMA_2_7B_32K/")
async def generate_text_HuggingFace_LLaMA_2_7B_32K(input: dict):
    """
    {"text": "where is capital of Iran"}
    
    """
    tokenizer = AutoTokenizer.from_pretrained("OpenAssistant/llama2-13b-orca-8k-3319", use_fast=False)
    model = AutoModelForCausalLM.from_pretrained("OpenAssistant/llama2-13b-orca-8k-3319", torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")

    system_message = "Read This RFP document and return title, summary, dates, and list of requirements"

    # Extract the text from the input JSON
    user_prompt = input.get('text')

    # Format the input for the model
    prompt = f"""{system_message}</s>{user_prompt}</s>"""
    inputs = tokenizer(prompt, return_tensors="pt")

    # Make sure to send the tensors to the same device where your model is
    inputs = {name: tensor.to("cuda") for name, tensor in inputs.items()}

    # Generate a response
    output = model.generate(**inputs, do_sample=True, top_p=0.95, top_k=0, max_new_tokens=256)
    output_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Return the generated text
    return {"generated_text": output_text}
#####################################HuggingFace LLaMA-2-7B-32K

#################################### TIKA Folder
import os
import glob
from tika import parser
from typing import List
 
def save_text_to_file(pdf_path: str, text: str):
    # Extract the name of the PDF without the extension
    file_name = os.path.splitext(pdf_path)[0]
    # Create a text file with the same name as the PDF and save the text content
    with open(f"{file_name}.txt", "w", encoding="utf-8") as text_file:
        text_file.write(text)

@app.post("/convert_pdf_to_txt/")
async def convert_pdf_to_txt(directory: str):
    # Check if the provided path is a valid directory
    if not os.path.isdir(directory):
        raise HTTPException(status_code=400, detail="Invalid directory.")

    subdirectories = [x[0] for x in os.walk(directory)]
    total_subdirs = len(subdirectories)
    print(f"Found {total_subdirs} subdirectories under {directory}")

    total_pdfs = 0
    for subdir in subdirectories:
        pdf_files = glob.glob(os.path.join(subdir, "*.pdf"))
        total_pdfs += len(pdf_files)
    print(f"Detected {total_pdfs} PDF files in all subdirectories.")

    for idx, subdir in enumerate(subdirectories, start=1):
        # Get all PDF files in the subdirectory
        pdf_files = glob.glob(os.path.join(subdir, "*.pdf"))
        print(f"Processing {len(pdf_files)} PDF files in {subdir}")
        for pdf_file in pdf_files:
            # Read the content of the PDF using Tika
            raw_text = parser.from_file(pdf_file)
            text_content = raw_text['content']
            # If text_content is None, replace it with an empty string
            if text_content is None:
                print(f"No extractable text found in {pdf_file}")
                text_content = ""
            # Save the text content into a text file with the same name as the PDF
            save_text_to_file(pdf_file, text_content)
            print(f"Converted to text")
        print(f"Subdirectory progress: {idx}/{total_subdirs} done!")

    return {"message": "Text extracted from PDF and saved successfully!"}



####################################TIKA Folder


#########################################TIKA

 
from typing import List
from fastapi import UploadFile

@app.post("/uploadpdf/")
async def upload_pdf_and_save_text_TIKA(pdf_files: List[UploadFile]=File(...)):
    for pdf_file in pdf_files:
        # Read the content of the PDF using Tika
        raw_text = parser.from_file(pdf_file.file)
        text_content = raw_text['content']

        # Save the text content into a text file with the same name as the PDF
        save_text_to_file(pdf_file.filename, text_content)

    return {"message": "Text extracted from PDF and saved successfully!"}

#########################################TIKA
###########################################togethercomputer/LLaMA-2-7B-32K simple prompt

class Item(BaseModel):
    prompt: str
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float

@app.post("/togethercomputer/LLaMA-2-7B-32K")
async def get_inference(prompt: str):
    """
Together AI: Together AI is a pioneering AI company committed to pushing the boundaries of AI through open-source research, models, and datasets. They provide decentralized cloud services to allow developers and researchers to train, fine-tune, and deploy generative AI models. Their transparent approach ensures you know how your model is trained and what data is used, you have full control over model customization, and that privacy is a priority as data storage is in your hands.
LLaMA-2-7B-32K:LLaMA-2-7B-32K, by Together AI, is an advanced language model that excels in handling long context lengths of up to 32K. This is achieved by using Meta's Llama-2 7B model as a base and extending it with position interpolation. The model has been trained using a balanced mixture of data sources and includes methods for fine-tuning for specific applications. Notably, this model is supported by software updates to ensure efficient inference and fine-tuning. Like all language models, users should be aware of potential biases and inaccuracies.
    """
    url = "https://api.together.xyz/inference"
    payload = {
        "model": "togethercomputer/LLaMA-2-7B-32K",
        "prompt": "What is the capital of Canada?",
        "max_tokens": 32,
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "repetition_penalty": 1
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": "Bearer c6a988b9f00ae9afc1aafeef26677eff7c7df4b2f0d3e25c43d77828f26ba249"
    }
    response = requests.post(url, json=payload, headers=headers)
    return response.text
###########################################togethercomputer/LLaMA-2-7B-32K RFP PDF
###########################################togethercomputer/LLaMA-2-7B-32K simple prompt

def llm_LLaMA_2_7B_32K(prompt):
    url = "https://api.together.xyz/inference"
    payload = {
        "model": "togethercomputer/LLaMA-2-7B-32K",
        "prompt": prompt,
        "max_tokens": 32000,
        "temperature": 0.7,
        "top_p": 0.7,
        "top_k": 50,
        "repetition_penalty": 1
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": "Bearer c6a988b9f00ae9afc1aafeef26677eff7c7df4b2f0d3e25c43d77828f26ba249"
    }
    response = requests.post(url, json=payload, headers=headers)
    
    # Check if the request was successful
    if response.status_code != 200:
        print(f"Request to API failed with status code {response.status_code}. The response is: {response.text}")
        return None

    # Check if the response can be decoded as JSON
    try:
        response_json = response.json()
    except ValueError:
        print(f"Cannot decode response as JSON. The response is: {response.text}")
        return None

    output = response_json['output']['choices'][0]['text']
    return output

def extract_text_from_pdf(file_as_bytes):
    pdf = PdfReader(file_as_bytes)
    text = ""
    for page in range(len(pdf.pages)):

        text += pdf.pages[page].extract_text()
    return text

def generate_token_chunks(tokens, chunk_size):
    return [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]

@app.get("/pdf/token_and_chunk_count")
async def get_token_and_chunk_count(pdf_url: str):
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()

        file_as_bytes = BytesIO(response.content)
        text = extract_text_from_pdf(file_as_bytes)

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokens = tokenizer.encode(text)
        token_count = len(tokens)
        chunks = generate_token_chunks(tokens,32000)
        chunk_count = len(chunks)

        return {"token_count": token_count, "chunk_count": chunk_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pdf/analyze_LLm_LLaMA_2_7B_32K")
async def analyze_LLm_LLaMA_2_7B_32K(pdf_url: str=default_file_path):
    """
    https://huggingface.co/togethercomputer/LLaMA-2-7B-32K?text=My+name+is+Lewis+and+I+like+to
Together AI: Together AI is a pioneering AI company committed to pushing the boundaries of AI through open-source research, models, and datasets. They provide decentralized cloud services to allow developers and researchers to train, fine-tune, and deploy generative AI models. Their transparent approach ensures you know how your model is trained and what data is used, you have full control over model customization, and that privacy is a priority as data storage is in your hands.
LLaMA-2-7B-32K:LLaMA-2-7B-32K, by Together AI, is an advanced language model that excels in handling long context lengths of up to 32K. This is achieved by using Meta's Llama-2 7B model as a base and extending it with position interpolation. The model has been trained using a balanced mixture of data sources and includes methods for fine-tuning for specific applications. Notably, this model is supported by software updates to ensure efficient inference and fine-tuning. Like all language models, users should be aware of potential biases and inaccuracies.
    """
    response = requests.get(pdf_url)
    response.raise_for_status()

    file_as_bytes = BytesIO(response.content)
    text = extract_text_from_pdf(file_as_bytes)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokens = tokenizer.encode(text)
    chunks = generate_token_chunks(tokens,16000)

    for i, token_chunk in enumerate(chunks):
        chunk = tokenizer.decode(token_chunk)
        if i == len(chunks) - 1:  # run for the last chunk
            title_prompt = f"Could you help me extract the title from this text?\n\n{chunk}"
            summary_prompt = f"Can you provide a summary of this text?\n\n{chunk}"
            due_date_prompt = f"Can you identify the due date mentioned in this text?\n\n{chunk}"
            requirements_prompt = f"What are the main requirements listed in this text?\n\n{chunk}"

            title = llm_LLaMA_2_7B_32K(title_prompt)
            summary = llm_LLaMA_2_7B_32K(summary_prompt)
            due_date = llm_LLaMA_2_7B_32K(due_date_prompt)
            requirements = llm_LLaMA_2_7B_32K(requirements_prompt)

            result = {
                'title': title,
                'summary': summary,
                'due_date': due_date,
                'requirements': requirements
            }
        else:  # for the first chunks, just instruct the model to read
            instruction = f"Please read this part of the text carefully:\n\n{chunk}"
            llm_LLaMA_2_7B_32K(instruction)

    return result  # return the result for the last chunk
###########################################togethercomputer/LLaMA-2-7B-32K RFP PDF



def llm(prompt):
    model_id = "gpt-3.5-turbo-16k"
    response = openai.ChatCompletion.create(
        model=model_id,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content']

def extract_text_from_pdf(file_as_bytes):
    pdf = PdfReader(file_as_bytes)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

def generate_token_chunks(tokens, chunk_size):
    """Breaks the tokens into chunks of the specified size."""
    return [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]

# @app.post("/pdf/analyze_gpt2")
# async def analyze_pdf_gpt2(pdf_url: str = default_file_path):
#     print("Fetching the PDF document...")
#     response = requests.get(pdf_url)
#     response.raise_for_status()

#     print("PDF document fetched. Converting PDF to text...")
#     file_as_bytes = BytesIO(response.content)
#     rfp_text = extract_text_from_pdf(file_as_bytes)

#     print("PDF converted to text. Saving text content to a .txt file...")
#     with open('rfp_text_content.txt', 'w', encoding='utf-8') as f:
#         f.write(rfp_text)

#     print("Saved text content to a .txt file.")
    
#     print("Calculating token count and chunk count...")
#     tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
#     tokens = tokenizer.encode(rfp_text)
#     token_count = len(tokens)
#     chunks = generate_token_chunks(tokens, 16000)
#     chunk_count = len(chunks)

#     print(f"Token count: {token_count}")
#     print(f"Chunk count: {chunk_count}")

#     print("Sending chunks to the model...")
#     for i, token_chunk in enumerate(chunks):
#         chunk = tokenizer.decode(token_chunk)
#         if i == len(chunks) - 1:  # run for the last chunk
#             print(f"Last chunk ({i+1}/{len(chunks)}). Generating prompts to extract specific information...")
#             title_prompt = f"Could you help me extract the title from this RFP?\n\n{chunk}"
#             summary_prompt = f"Can you provide a summary of this RFP?\n\n{chunk}"
#             due_date_prompt = f"Can you identify the due date mentioned in this RFP?\n\n{chunk}"
#             requirements_prompt = f"What are the main requirements listed in this RFP?\n\n{chunk}"

#             print("Sending prompts to the model...")
#             title = llm(title_prompt)
#             summary = llm(summary_prompt)
#             due_date = llm(due_date_prompt)
#             requirements = llm(requirements_prompt)

#             print("Prompts processed. Compiling the results...")
#             result = {
#                 'title': title,
#                 'summary': summary,
#                 'due_date': due_date,
#                 'requirements': requirements
#             }
#             print("Results compiled.")
#         else:  # for the first chunks, just instruct the model to read
#             print(f"Chunk {i+1}/{len(chunks)}. Instructing the model to read the chunk...")
#             instruction = f"Please read this part of the RFP carefully:\n\n{chunk}"
#             llm(instruction)

#     print("Returning the results.")
#     return result  # return the result for the last chunk


default_file_path = 'https://rfp-cases.s3.ca-tor.cloud-object-storage.appdomain.cloud/2023/Bell/C09AD188-0000-C922-B40E-E6AA3F3B9A1F/906090ae-344d-4108-a5d8-9ca42f3fc292/906090ae-344d-4108-a5d8-9ca42f3fc292.pdf'

load_dotenv()  # take environment variables from .env.
openai.api_key = os.getenv("OPENAI_API_KEY")

import requests


###########################################################################################StableBeluga2
API_URL_StableBeluga2 = "https://api-inference.huggingface.co/models/stabilityai/StableBeluga2"
headers_StableBeluga2  = {"Authorization": "Bearer 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"}

def query_stablebeluga2(payload):
    response = requests.post(API_URL_StableBeluga2, headers=headers_StableBeluga2, json=payload)
    return response.json()

@app.post("/pdf/analyze_stablebeluga2")
async def analyze_pdf_stablebeluga2(pdf_url: str = default_file_path):
    """
    Title: Harnessing the Hugging Face Inference API for Running Stable Beluga 2 Model

    Introduction:

    The Stable Beluga 2 model, developed by Stability AI, is an autoregressive language model fine-tuned on a Llama2 70B model and trained on an Orca style Dataset. It excels at understanding and generating text and can be instructed to perform tasks it hasn't been explicitly trained for by casting them as text generation tasks.

    Preparation:

    To get started with the API, you need to register or login and obtain a User Access or API token from your Hugging Face profile settings. This token is necessary for running inference on your private models and should look like this: hf_xxxxx.

    Running Inference:

    Once you have chosen the Stable Beluga 2 model from the Model Hub, you can run inference using the following code, where <API_TOKEN> should be replaced with your personal API token:

    Note that the API is continuously updated, so always check the documentation for the latest information. For further inquiries, contact api-enterprise@huggingface.co.
    """
    with open('rfp_.txt', 'r', encoding='utf-8') as file:
        rfp_text = file.read()
    print("load brief rfp text")
    
    print("Calculating token count and chunk count...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokens = tokenizer.encode(rfp_text)
    token_count = len(tokens)
    chunks = generate_token_chunks(tokens,2000)
    chunk_count = len(chunks)

    print(f"Token count: {token_count}")
    print(f"Chunk count: {chunk_count}")

    print("Sending chunks to the model...")
    for i, token_chunk in enumerate(chunks):
        chunk = tokenizer.decode(token_chunk)
        if i == len(chunks) - 1:  # run for the last chunk
            print(f"Last chunk ({i+1}/{len(chunks)}). Generating prompts to extract specific information...")
            title_prompt = f"### System:\nYou are Stable Beluga, an AI that follows instructions extremely well. Help as much as you can. Remember, be safe, and don't do anything illegal.\n\n### User:\nCould you help me extract the title from this RFP?\n\n### Assistant:\n\n{chunk}"
            summary_prompt = f"### System:\nYou are Stable Beluga, an AI that follows instructions extremely well. Help as much as you can. Remember, be safe, and don't do anything illegal.\n\n### User:\nCan you provide a summary of this RFP?\n\n### Assistant:\n\n{chunk}"
            due_date_prompt = f"### System:\nYou are Stable Beluga, an AI that follows instructions extremely well. Help as much as you can. Remember, be safe, and don't do anything illegal.\n\n### User:\nCan you identify the due date mentioned in this RFP?\n\n### Assistant:\n\n{chunk}"
            requirements_prompt = f"### System:\nYou are Stable Beluga, an AI that follows instructions extremely well. Help as much as you can. Remember, be safe, and don't do anything illegal.\n\n### User:\nWhat are the main requirements listed in this RFP?\n\n### Assistant:\n\n{chunk}"

            print("Sending prompts to the model...")
            title = query_stablebeluga2({"inputs": title_prompt})
            summary = query_stablebeluga2({"inputs": summary_prompt})
            due_date = query_stablebeluga2({"inputs": due_date_prompt})
            requirements = query_stablebeluga2({"inputs": requirements_prompt})

            print("Prompts processed. Compiling the results...")
            result = {
                'title': title,
                'summary': summary,
                'due_date': due_date,
                'requirements': requirements
            }
            print("Results compiled.")
        else:  # for the first chunks, just instruct the model to read
            print(f"Chunk {i+1}/{len(chunks)}. Instructing the model to read the chunk...")
            instruction = f"### System:\nYou are Stable Beluga, an AI that follows instructions extremely well. Help as much as you can. Remember, be safe, and don't do anything illegal.\n\n### User:\nPlease read this part of the RFP carefully:\n\n### Assistant:\n\n{chunk}"
            query_stablebeluga2({"inputs": instruction})

    print("Returning the results.")
    return result  # return the result for the last chunk

###########################################################################################StableBeluga2

##########################################################################################Bloom
API_URL = "https://api-inference.huggingface.co/models/bigscience/bloom"
headers = {"Authorization": "Bearer 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"}

def query_bigscience_bloom(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

@app.post("/pdf/analyze_bigscience_bloom")
async def analyze_pdf_bigscience_bloom(pdf_url: str = default_file_path):
    """
    Title: Harnessing the Hugging Face Inference API for Running BigScience's BLOOM Model

Introduction:

The 🤗 Hosted Inference API is an incredibly powerful and versatile tool that lets you leverage over 80,000 Transformer models, including T5, Blenderbot, Bart, GPT-2, Pegasus, and more. In addition, you can upload, manage, and serve your own models privately, running tasks ranging from classification and NER to conversational, summarization, translation, question-answering, and embeddings extraction tasks.

This document provides a brief guide on how to use this API to run BigScience's BLOOM model, a large language model proficient in 46 languages and 13 programming languages.

Preparation:

To get started with the API, you need to register or login and obtain a User Access or API token from your Hugging Face profile settings. This token is necessary for running inference on your private models and should look like this: hf_xxxxx.

Running Inference:

Once you have chosen the BLOOM model from the Model Hub, you can run inference using the following code, where <API_TOKEN> should be replaced with your personal API token:
 

Depending on the task (aka pipeline) the model is configured for, the request will accept specific parameters. When sending requests to run any model, API options allow you to specify the caching and model loading behavior. All API options and parameters are detailed here [detailed_parameters] (Please replace this placeholder with the appropriate link).

Using CPU-Accelerated Inference:

For API customers, your token automatically enables CPU-Accelerated inference if the model type supports it, which results in a ~10x speedup in inference times. To verify you're using the CPU-Accelerated version of a model, check the x-compute-type header of your requests, which should be cpu+optimized.

Model Loading and Latency:

The Hosted Inference API can serve predictions on-demand from over 100,000 models deployed on the Hugging Face Hub, dynamically loaded on shared infrastructure. If the model isn't loaded in memory, the API will return a 503 response. For large volume or predictable latencies, consider using the paid solution Inference Endpoints.

About BigScience's BLOOM:

BLOOM, developed by BigScience, is an autoregressive Large Language Model (LLM) trained on vast amounts of text data using industrial-scale computational resources. It's capable of outputting coherent text in 46 languages and 13 programming languages, hardly distinguishable from human-written text. The model can be instructed to perform tasks it hasn't been explicitly trained for by casting them as text generation tasks. The model consists of 176,247,271,424 parameters, 70 layers, and 112 attention heads.

Note that the API is continuously updated, so always check the documentation for the latest information. For further inquiries, contact api-enterprise@huggingface.co.
    """
    # print("Fetching the PDF document...")
    # response = requests.get(pdf_url)
    # response.raise_for_status()

    # print("PDF document fetched. Converting PDF to text...")
    # file_as_bytes = BytesIO(response.content)
    # rfp_text = extract_text_from_pdf(file_as_bytes)

    # print("PDF converted to text. Saving text content to a .txt file...")
    # with open('rfp_text_content.txt', 'w', encoding='utf-8') as f:
    #     f.write(rfp_text)

    # print("Saved text content to a .txt file.")
    with open('rfp_.txt', 'r', encoding='utf-8') as file:
        rfp_text = file.read()
    print("load brief rfp text")
    
    print("Calculating token count and chunk count...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokens = tokenizer.encode(rfp_text)
    token_count = len(tokens)
    chunks = generate_token_chunks(tokens,2000)
    chunk_count = len(chunks)

    print(f"Token count: {token_count}")
    print(f"Chunk count: {chunk_count}")

    print("Sending chunks to the model...")
    for i, token_chunk in enumerate(chunks):
        chunk = tokenizer.decode(token_chunk)
        if i == len(chunks) - 1:  # run for the last chunk
            print(f"Last chunk ({i+1}/{len(chunks)}). Generating prompts to extract specific information...")
            title_prompt = f"Could you help me extract the title from this RFP?\n\n{chunk}"
            summary_prompt = f"Can you provide a summary of this RFP?\n\n{chunk}"
            due_date_prompt = f"Can you identify the due date mentioned in this RFP?\n\n{chunk}"
            requirements_prompt = f"What are the main requirements listed in this RFP?\n\n{chunk}"

            print("Sending prompts to the model...")
            title = query_bigscience_bloom({"inputs": title_prompt})
            summary = query_bigscience_bloom({"inputs": summary_prompt})
            due_date = query_bigscience_bloom({"inputs": due_date_prompt})
            requirements = query_bigscience_bloom({"inputs": requirements_prompt})

            print("Prompts processed. Compiling the results...")
            result = {
                'title': title,
                'summary': summary,
                'due_date': due_date,
                'requirements': requirements
            }
            print("Results compiled.")
        else:  # for the first chunks, just instruct the model to read
            print(f"Chunk {i+1}/{len(chunks)}. Instructing the model to read the chunk...")
            instruction = f"Please read this part of the RFP carefully:\n\n{chunk}"
            query_bigscience_bloom({"inputs": instruction})

    print("Returning the results.")
    return result  # return the result for the last chunk


################################################################################### gpt-3.5-turbo-16k
def llm(prompt):
    model_id = "gpt-3.5-turbo-16k"
    response = openai.ChatCompletion.create(
        model=model_id,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content']

def extract_text_from_pdf(file_as_bytes):
    pdf = PdfReader(file_as_bytes)
    text = ""
    for page in range(len(pdf.pages)):

        text += pdf.pages[page].extract_text()
    return text

def generate_token_chunks(tokens, chunk_size):
    """Breaks the tokens into chunks of the specified size."""
    return [tokens[i:i+chunk_size] for i in range(0, len(tokens), chunk_size)]

@app.get("/pdf/token_and_chunk_count")
async def get_token_and_chunk_count(pdf_url: str = default_file_path):
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()

        file_as_bytes = BytesIO(response.content)
        text = extract_text_from_pdf(file_as_bytes)

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokens = tokenizer.encode(text)
        token_count = len(tokens)
        chunks = generate_token_chunks(tokens,16000)
        chunk_count = len(chunks)

        return {"token_count": token_count, "chunk_count": chunk_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pdf/analyze_gpt3_5_turbo_16k")
async def analyze_pdf_gpt3_5_turbo_16k(pdf_url: str = default_file_path):
    print("Fetching the PDF document...")
    response = requests.get(pdf_url)
    response.raise_for_status()

    print("PDF document fetched. Converting PDF to text...")
    file_as_bytes = BytesIO(response.content)
    rfp_text = extract_text_from_pdf(file_as_bytes)

    print("PDF converted to text. Saving text content to a .txt file...")
    with open('rfp_text_content.txt', 'w', encoding='utf-8') as f:
        f.write(rfp_text)

    print("Saved text content to a .txt file.")
    
    print("Calculating token count and chunk count...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokens = tokenizer.encode(rfp_text)
    token_count = len(tokens)
    chunks = generate_token_chunks(tokens,16000)
    chunk_count = len(chunks)

    print(f"Token count: {token_count}")
    print(f"Chunk count: {chunk_count}")

    print("Sending chunks to the model...")
    for i, token_chunk in enumerate(chunks):
        chunk = tokenizer.decode(token_chunk)
        if i == len(chunks) - 1:  # run for the last chunk
            print(f"Last chunk ({i+1}/{len(chunks)}). Generating prompts to extract specific information...")
            title_prompt = f"Could you help me extract the title from this RFP?\n\n{chunk}"
            summary_prompt = f"Can you provide a summary of this RFP?\n\n{chunk}"
            due_date_prompt = f"Can you identify the due date mentioned in this RFP?\n\n{chunk}"
            requirements_prompt = f"What are the main requirements listed in this RFP?\n\n{chunk}"

            print("Sending prompts to the model...")
            title = llm(title_prompt)
            summary = llm(summary_prompt)
            due_date = llm(due_date_prompt)
            requirements = llm(requirements_prompt)

            print("Prompts processed. Compiling the results...")
            result = {
                'title': title,
                'summary': summary,
                'due_date': due_date,
                'requirements': requirements
            }
            print("Results compiled.")
        else:  # for the first chunks, just instruct the model to read
            print(f"Chunk {i+1}/{len(chunks)}. Instructing the model to read the chunk...")
            instruction = f"Please read this part of the RFP carefully:\n\n{chunk}"
            llm(instruction)

    print("Returning the results.")
    return result  # return the result for the last chunk
################################################################################### gpt-3.5-turbo-16k
################################################################################### Get list of watsonx.ai models and descriptions
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.foundation_models import Model

my_credentials = {
    "url": "https://us-south.ml.cloud.ibm.com",
    "apikey": "PhZAweINcKPpv_Gxkx0nlJO5SYilql1ggBl5BblhEp3J"
}
gen_parms = None
project_id = "7c0bd14d-79a7-416b-b0af-cac2bfc41412"
space_id = None
verify = False
@app.get("/models", tags=["List available models and details"])
async def get_models():
    models = {}
    for model_name in ModelTypes._member_names_:
        model_id = getattr(ModelTypes, model_name)
        model = Model(model_id, my_credentials, gen_parms, project_id, space_id, verify)
        model_details = model.get_details()
        models[model_name] = model_details

    return models

################################################################################### Get list of watsonx.ai models and descriptions




###################################################################tags=["Step 2"]  WatsonX.ai analyze_pdf 
from enum import Enum

class ModelName(str, Enum):
    google_flan_ul2 = "google/flan-ul2"
    google_flan_t5_xxl = "google/flan-t5-xxl"
    bigscience_mt0_xxl = "bigscience/mt0-xxl"
    eleutherai_gpt_neox_20b = "eleutherai/gpt-neox-20b"
    ibm_mpt_7b_instruct2 = "ibm/mpt-7b-instruct2"

@app.post("/pdf/analyze_pdf_Watsonx_ai_step_2", tags=["Step 2 Extract Title, summary and date"])
async def analyze_pdf_Watsonx_ai_step_2(
    pdf_url: str = default_file_path,
    selected_model: ModelName = ModelName.google_flan_ul2
):
    model_id = selected_model.value
    api_key = "PhZAweINcKPpv_Gxkx0nlJO5SYilql1ggBl5BblhEp3J"
    

    if selected_model not in selected_model:
        return {"error": "Selected model is not in the available models list."}

    api_key = "PhZAweINcKPpv_Gxkx0nlJO5SYilql1ggBl5BblhEp3J"

    access_token = obtain_access_token(api_key)
    project_id = "7c0bd14d-79a7-416b-b0af-cac2bfc41412"
    
    parameters = {
        "decoding_method": "greedy",
        "max_new_tokens": 1024,
        "repetition_penalty": 2
    }
    numberOfChunks = 1 

    prompt = Prompt(access_token, project_id)

    response = requests.get(pdf_url)
    response.raise_for_status()

    file_as_bytes = BytesIO(response.content)
    rfp_text = extract_text_from_pdf_xai(file_as_bytes)

    chunks = generate_chunks_xai(rfp_text)

    title = []
    summary = []
    due_date = []
    date_desc=[]
    summary_clean = []  # List to hold cleaned summaries

    for i, chunk in enumerate(chunks):
        if numberOfChunks > 0 and i >= numberOfChunks:
            break
        print("chunk size is =",len(chunk),"/",len(chunks))
        # print("chunk =",chunk)
        #my first prompt
        title_prompt = f"Extract the title of the following RFP document. Read it carefully and identify the most suitable title for this document, ensuring that it accurately represents the content. Only include English content:\n\nDocument: \n###\n{chunk}\n###\n\nTitle: "
        summary_prompt = f"Please provide a concise and clear summary of the following document, focusing on the key details and eliminating any redundant or unnecessary information:\n\nDocument: \n###\n{chunk}\n###\n\nSummary: "

        date_prompt = f"find any date in the document,it is a RFP document, read it carefully and extract any date in it:\n\nDocument: \n###\n{chunk}\n###\n\nDue Date: "
        date_desc_prompt = f"find any date and its description in the document,it is a RFP document, read it carefully and extract any date and  description related to that date:\n\nDocument: \n###\n{chunk}\n###\n\nDue Date: "


        # #chatgpt suggeted prompt:
        # title_prompt = f"Extract the title of the following document, it is an RFP document, read it carefully and extract the best title possible for it:\n\nDocument: \n###\n{chunk}\n###\n\nTitle: "
        # summary_prompt = f"it is an RFP document, read it carefully, find the summary part in English, clean it, remove extra and irrelevant parts of it:\n\nDocument: \n###\n{chunk}\n###\n\nSummary: "
        # date_prompt = f"find any date in the document, it is an RFP document, read it carefully and extract any date related to the closing date of the bid:\n\nDocument: \n###\n{chunk}\n###\n\nDue Date: "
        # date_desc_prompt = f"find any date and its description in the document, it is an RFP document, read it carefully and extract any date and description related to the closing date of the bid:\n\nDocument: \n###\n{chunk}\n###\n\nDue Date: "

        title_chunk = prompt.generate(title_prompt, selected_model, parameters)
        title.append(title_chunk)
        summary_chunk = prompt.generate(summary_prompt, selected_model, parameters)
        summary.append(summary_chunk)
        Date_chunk = prompt.generate(date_prompt, selected_model, parameters)
        due_date.append(Date_chunk)
        Date_Desc_chunk = prompt.generate(date_desc_prompt, selected_model, parameters)
        date_desc.append(Date_Desc_chunk)
        # New code: Sending summary back to model for post-processing
        clean_summary_prompt = f"The following text may contain redundant or unclear information. Please rewrite it into a clean, well-structured, and succinct summary. Remove none characters. Only english. only meaningfull words :\n\nSummary: \n###\n{summary_chunk}\n###\n\nClean Summary: "
        clean_summary_chunk = prompt.generate(clean_summary_prompt, selected_model, parameters)
        summary_clean.append(clean_summary_chunk)

    return {"title": title, "summary": summary, "summary_clean": summary_clean, "date": due_date, "date description": date_desc }  # Updated 'summary' to 'summary_clean'



@app.post("/pdf/analyze_pdf_Watsonx_ai_flan_ul2_v1", tags=["Step 2  Extract Title, summary and date"])
async def analyze_pdf_Watsonx_ai_flan_ul2(pdf_url: str = default_file_path):
    api_key = "PhZAweINcKPpv_Gxkx0nlJO5SYilql1ggBl5BblhEp3J"

    access_token = obtain_access_token(api_key)
    project_id = "7c0bd14d-79a7-416b-b0af-cac2bfc41412"
    model_id = "google/flan-ul2"
    parameters = {
        "decoding_method": "greedy",
        "max_new_tokens": 1024,
        "repetition_penalty": 2
    }
    numberOfChunks = 1 

    prompt = Prompt(access_token, project_id)

    response = requests.get(pdf_url)
    response.raise_for_status()

    file_as_bytes = BytesIO(response.content)
    rfp_text = extract_text_from_pdf_xai(file_as_bytes)

    chunks = generate_chunks_xai(rfp_text)

    title = []
    summary = []
    due_date = []
    date_desc=[]
    for i, chunk in enumerate(chunks):
        if numberOfChunks > 0 and i >= numberOfChunks:
            break
        print("chunk size is =",len(chunk),"/",len(chunks))
        # print("chunk =",chunk)
        #my first prompt
        title_prompt = f"Extract the title of the following document, it is a RFP document, read it carefully and extract the best title possible for it:\n\nDocument: \n###\n{chunk}\n###\n\nTitle: "
        summary_prompt = f"it is a RFP document, read it carefully, find summary part in english, clean it, remove extra and unrelavant part of it  :\n\nDocument: \n###\n{chunk}\n###\n\nSummary: "
        date_prompt = f"find any date in the document,it is a RFP document, read it carefully and extract any date in it:\n\nDocument: \n###\n{chunk}\n###\n\nDue Date: "
        date_desc_prompt = f"find any date and its description in the document,it is a RFP document, read it carefully and extract any date and  description related to that date:\n\nDocument: \n###\n{chunk}\n###\n\nDue Date: "

        # #chatgpt suggeted prompt:
        # title_prompt = f"Extract the title of the following document, it is an RFP document, read it carefully and extract the best title possible for it:\n\nDocument: \n###\n{chunk}\n###\n\nTitle: "
        # summary_prompt = f"it is an RFP document, read it carefully, find the summary part in English, clean it, remove extra and irrelevant parts of it:\n\nDocument: \n###\n{chunk}\n###\n\nSummary: "
        # date_prompt = f"find any date in the document, it is an RFP document, read it carefully and extract any date related to the closing date of the bid:\n\nDocument: \n###\n{chunk}\n###\n\nDue Date: "
        # date_desc_prompt = f"find any date and its description in the document, it is an RFP document, read it carefully and extract any date and description related to the closing date of the bid:\n\nDocument: \n###\n{chunk}\n###\n\nDue Date: "

        title_chunk = prompt.generate(title_prompt, model_id, parameters)
        title.append(title_chunk)
        summary_chunk = prompt.generate(summary_prompt, model_id, parameters)
        summary.append(summary_chunk)
        Date_chunk = prompt.generate(date_prompt, model_id, parameters)
        due_date.append(Date_chunk)
        Date_Desc_chunk = prompt.generate(date_desc_prompt, model_id, parameters)
        date_desc.append(Date_Desc_chunk)

    return {"title": title, "summary": summary, "date": due_date,"date description": date_desc, "Raw Chunk": chunk}


# Function to get Access Token
def obtain_access_token(api_key):
    """
    This function takes an IBM Cloud API Key and returns an access token.
    """
    url = "https://iam.cloud.ibm.com/identity/token"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json"
    }
    data = {
        "apikey": api_key,
        "grant_type": "urn:ibm:params:oauth:grant-type:apikey"
    }
    response = requests.post(url, data=data, headers=headers)
    if response.status_code == 200:
        return response.json()["access_token"]
    else:
        raise Exception("Failed to obtain access token: " + response.text)

# Extract text from a PDF file
def extract_text_from_pdf_xai(file_bytes):
    """
    This function takes a PDF file in the form of bytes, 
    reads the PDF file and returns the extracted text.
    """
    pdf_file = PdfReader(file_bytes)
    text = ""
    for page in pdf_file.pages:
        text += page.extract_text()
    return text

# Define a class for generating prompts using the Watson Machine Learning API
class Prompt:
    """
    This class is used for generating prompts using the Watson Machine Learning API.
    It needs an access token and a project ID for initialization.
    """
    def __init__(self, access_token, project_id):
        self.access_token = access_token
        self.project_id = project_id

    def generate(self, input, model_id, parameters):
        """
        This function takes an input prompt, a model ID, and parameters for the model,
        sends a request to the Watson Machine Learning API, and returns the generated text.
        """
        wml_url = "https://us-south.ml.cloud.ibm.com/ml/v1-beta/generation/text?version=2023-05-28"
        headers = {
            "Authorization": "Bearer " + self.access_token,
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        data = {
            "model_id": model_id,
            "input": input,
            "parameters": parameters,
            "project_id": self.project_id
        }
        response = requests.post(wml_url, json=data, headers=headers)
        if response.status_code == 200:
            return response.json()["results"][0]["generated_text"]
        else:
            return response.text

# Define a function for splitting text into chunks
def generate_chunks_xai(text, max_chunk_size=20000):
    """
    This function takes a text and a maximum chunk size, 
    splits the text into chunks of the specified size and returns the chunks.
    """
    return [text[i:i+max_chunk_size] for i in range(0, len(text), max_chunk_size)]

# Define a function for saving text as a PDF file
def save_text_as_pdf_xai(text, filename):
    """
    This function takes a text and a filename, 
    creates a new PDF file with the specified filename and writes the text into the PDF file.
    """
    c = canvas.Canvas(filename)
    for i, line in enumerate(text.splitlines()):
        c.drawString(10, 800-i*15, line)  # adjust the numbers to suit your needs
    c.save()

################################################################### WatsonX.ai analyze_pdf_Watsonx_flan_ul2
 















class RFPDetails(BaseModel):
    title: str
    summary: str
    due_date: Optional[str]
    methods_of_submission: str

class Requirement(BaseModel):
    id: int
    description: str

class RequirementDetails(BaseModel):
    requirements: List[Requirement]

class ClassifiedRequirements(BaseModel):
    classified_requirements: List[dict]

class EnrichmentRequirement(BaseModel):
    requirements: List[Requirement]
    questions: List[str] = []
    uncertainties: List[str] = []


requirements = {
    "requirements": [
        {"id": 1, "description": "The software must include both English and French user interfaces without requiring any additional customization. The interface must allow each user to choose between English and French."},
        {"id": 2, "description": "The Contractor must provide web-based training (including training material) in English for Administrators and CRA users."},
        {"id": 3, "description": "The Contractor must provide manuals that define all functions and includes complete instructions for the operation of the product and are downloadable from the Internet."},
        {"id": 4, "description": "The software must create a bug report or incident in BMC Remedy."},
        {"id": 5, "description": "The software must create a bug report or incident in BMC Helix."},
        {"id": 6, "description": "The software must create a bug report or incident in Atlassian Jira Data Center."},
        {"id": 7, "description": "The software must be commercially available at the time of RFP closing within the product bid. Alpha or beta versions of the product will not be accepted."},
        {"id": 8, "description": "The software must run on-premises on a virtualized x64 architecture on Linux."},
        {"id": 9, "description": "The software must run on both Canadian Microsoft Azure and Canadian Amazon Web Services accessible through a service endpoint that is not accessible on the Internet such as VPC or VNET."},
        {"id": 10, "description": "The software must run on Red Hat Enterprise Linux (RHEL) v7 and subsequent versions on 64-bit microprocessor architectures for the duration of the contract."},
        {"id": 11, "description": "The software solution must limit system access to authorized users, processes acting on behalf of authorized users, and devices (including other systems) using replay-resistant authentication."},
        {"id": 12, "description": "The software must use LDAPv3 (Lightweight Directory Access Protocol version 3) for user account authentication. The software must employ LDAPS."},
        {"id": 13, "description": "The software must allow disabling automatic updates."},
        {"id": 14, "description": "The software must not interfere with the operation of any Anti-Virus, Anti-Malware, Data Loss Prevention, or Host Intrusion Detection systems operating on a host computer."},
        {"id": 15, "description": "The software must be compatible with internet browsers Google Chrome v83 and all subsequent releases, or Microsoft Edge v42 and all subsequent releases."},
        {"id": 16, "description": "The software must include a repository to store log data, alerts and dashboards, or must use one or more of the following DBMSs to create, maintain, and delete repositories: DB2 LUW; PostgreSQL; or Oracle."},
        {"id": 17, "description": "The repository used by the software must store a minimum of 20 TB of data."},
        {"id": 18, "description": "The product must operate on networks running IPv4."},
        {"id": 19, "description": "The product must operate on networks running IPv6."},
        {"id": 20, "description": "The software's user login interface used in remote credential authentication and authorization connections to the platform must be configurable to use TLS connections between server and client using TLS versions 1.2 or later."},
        {"id": 21, "description": "The software must use both TLS versions 1.2 or later for remote connections and must be configured to use only FIPS-based cipher suites recommended by the National Institute of Standards and Technology in NIST SP 800-52 Rev. 2 and by the Canadian Centre for Cyber Security."},
        {"id": 22, "description": "The software platform must be configured to use X.509 version 3 TLS certificates for mutual authentication between server and the client."},
        {"id": 23, "description": "The software must allow CRA to integrate existing credential access management systems through Federated Identity, leveraging SAML 2.0 (and subsequent versions)."},
        {"id": 24, "description": "The software must use file integrity verification mechanisms to detect unauthorized changes in its repository."},
        {"id": 25, "description": "The platform must include encryption capability of all data at rest when in storage using FIPS 140-2 compliant AES algorithm with at least one of the following AES key lengths: 128 bits; 192 bits; or 256 bits."},
        {"id": 26, "description": "The software must use either RESTful or SOAP APIs and be configurable to comply with the following GC Standards on API implementations."},
        {"id": 27, "description": "The software must be deployable in both production and testing environments across all IT platforms and used to monitor both testing and production environments."},
        {"id": 28, "description": "The software must permit deployment for up to 10 instances of the software (at the CRA's discretion) that are independent from the monitored environments, for the purpose of maintenance, testing and training of the Log Analytics software."},
        {"id": 29, "description": "The software must read log entries stored at different tiers based on age and criticality."},
        {"id": 30, "description": "The software must collect a minimum of 12 GB of log data per hour."},
        {"id": 31, "description": "The software must collect from logs residing on a minimum of 500 servers. Each server will have a minimum of one log."},
        {"id": 32, "description": "The software must ingest and interpret directly from the Mainframe all of the following Mainframe log types: a) SMF (System Management Facility); b) SYSLOG; c) DB2 on z/OS job log; d) MQ on z/OS job log; e) CICS Logs; and f) CICS Transaction Gateway Logs. For EACH listed item above, Bidders must identify whether their proposed software can meet each item out of the box or if configuration is required."},
        {"id": 33, "description": "The software must ingest and interpret directly from the server hosting the logs all of the following log types: a) MQ on Red Hat Linux (on x86) logs; b) Weblogic logs; c) Wildfly logs; d) Apache logs; e) LDAP logs; f) Active Directory; g) Centralized logging services managed in Microsoft Azure; h) Centralized logging services managed in Amazon Web Services; i) Siteminder Policy Server smps.log; j) Siteminder Policy Server smtrace*.log; and k) Oracle DB Audit logs. For EACH listed item above, Bidders must identify whether their proposed software can meet each item out of the box or if configuration is required."},
        {"id": 34, "description": "The software must accept as input and process (for alerts and to make searchable) log files up to and including 500GB in size."}
    ]
}

classified_requirements_list = [
    {"Software": "Software", "User Interfaces": "English and French user interfaces"},
    {"Contractor": "Contractor", "Training": "Provide web-based training"},
    {"Contractor": "Contractor", "Manuals": "Provide manuals"},
    {"Software": "Software", "Bug Reporting": "Create a bug report in BMC Remedy"},
    {"Software": "Software", "Bug Reporting": "Create a bug report in BMC Helix"},
    {"Software": "Software", "Bug Reporting": "Create a bug report in Atlassian Jira Data Center"},
    {"Software": "Software", "Product Version": "Be commercially available, alpha or beta versions not accepted"},
    {"Software": "Software", "Infrastructure": "Run on-premises on virtualized x64 architecture on Linux"},
    {"Software": "Software", "Infrastructure": "Run on both Canadian Microsoft Azure and Canadian Amazon Web Services"},
    {"Software": "Software", "Operating System": "Run on Red Hat Enterprise Linux (RHEL) v7 and subsequent versions"},
    {"Software": "Software", "Access Control": "Limit system access to authorized users"},
    {"Software": "Software", "Authentication": "Use LDAPv3 for user account authentication"},
    {"Software": "Software", "Update": "Allow disabling automatic updates"},
    {"Software": "Software", "Compatibility": "Not interfere with operation of Anti-Virus, Anti-Malware, Data Loss Prevention, or Host Intrusion Detection systems"},
    {"Software": "Software", "Browser Compatibility": "Be compatible with Google Chrome v83 and all subsequent releases, or Microsoft Edge v42 and all subsequent releases"},
    {"Software": "Software", "Database": "Include a repository or use DBMSs to store log data, alerts and dashboards"},
    {"Software": "Software", "Data Storage": "Store a minimum of 20 TB of data in repository"},
    {"Product": "Product", "Networking": "Operate on networks running IPv4"},
    {"Product": "Product", "Networking": "Operate on networks running IPv6"},
    {"Software": "Software", "Connection": "Use TLS versions 1.2 or later for remote connections"},
    {"Software": "Software", "Encryption": "Use both TLS versions 1.2 or later for remote connections and be configured to use FIPS-based cipher suites"},
    {"Software": "Software", "Authentication": "Use X.509 version 3 TLS certificates for mutual authentication"},
    {"Software": "Software", "Access Control": "Allow CRA to integrate existing credential access management systems"},
    {"Software": "Software", "Verification": "Use file integrity verification mechanisms"},
    {"Platform": "Platform", "Encryption": "Include encryption capability of all data at rest using FIPS 140-2 compliant AES algorithm"},
    {"Software": "Software", "APIs": "Use either RESTful or SOAP APIs"},
    {"Software": "Software", "Deployment": "Be deployable in both production and testing environments"},
    {"Software": "Software", "Deployment": "Permit deployment for up to 10 instances of the software"},
    {"Software": "Software", "Data Management": "Read log entries stored at different tiers based on age and criticality"},
    {"Software": "Software", "Data Collection": "Collect a minimum of 12 GB of log data per hour"},
    {"Software": "Software", "Data Collection": "Collect from logs residing on a minimum of 500 servers"},
    {"Software": "Software", "Log Types": "Ingest and interpret directly from the Mainframe various log types"},
    {"Software": "Software", "Log Types": "Ingest and interpret directly from the server hosting the logs various log types"},
    {"Software": "Software", "Data Input": "Accept as input and process log files up to and including 500GB in size"},
]
enriched_requirements = [
    {"id": 1, "category": "Software", "description": "The software should have user interfaces in both English and French, with no additional customization required. Users should be able to choose their preferred language."},
    {"id": 2, "category": "Software", "description": "The contractor should provide comprehensive web-based training in English, designed specifically for administrators and CRA users, including all necessary training materials."},
    {"id": 3, "category": "Software", "description": "The contractor is required to provide comprehensive manuals that clearly define all functions of the product and include complete instructions for its operation. These manuals must be available for download from the internet."},
    {"id": 4, "category": "Software", "description": "The software must include an automated system for creating a bug report or incident in BMC Remedy, to ensure efficient issue tracking and resolution."},
    {"id": 5, "category": "Software", "description": "The software must include an automated system for creating a bug report or incident in BMC Helix, to ensure efficient issue tracking and resolution."},
    {"id": 6, "category": "Software", "description": "The software must include an automated system for creating a bug report or incident in Atlassian Jira Data Center, ensuring efficient issue tracking and resolution."},
    {"id": 7, "category": "Software", "description": "The software provided must be commercially available, alpha or beta versions will not be accepted. This ensures a reliable, tested, and stable product."},
    {"id": 8, "category": "Software", "description": "The software must be capable of running on-premises on virtualized x64 architecture on Linux, providing flexibility and performance efficiency in deployment."},
    {"id": 9, "category": "Software", "description": "The software must be able to operate on both Canadian Microsoft Azure and Canadian Amazon Web Services, enabling versatile deployment across multiple cloud platforms."},
    {"id": 10, "category": "Software", "description": "The software must be compatible with Red Hat Enterprise Linux (RHEL) v7 and subsequent versions, ensuring its adaptability to various system environments."},
    {"id": 11, "category": "Software", "description": "The software must have robust access control mechanisms to limit system access to authorized users only, ensuring high security and privacy."},
    {"id": 12, "category": "Software", "description": "The software should use LDAPv3 for user account authentication, which provides robust security and seamless integration with many enterprise systems."},
    {"id": 13, "category": "Software", "description": "The software should have an option to disable automatic updates, allowing system administrators to manage updates as per their schedule and needs."},
    {"id": 14, "category": "Software", "description": "The software should not interfere with the operation of Anti-Virus, Anti-Malware, Data Loss Prevention, or Host Intrusion Detection systems, ensuring compatibility with essential security tools."},
    {"id": 15, "category": "Software", "description": "The software should be compatible with Google Chrome v83 and all subsequent releases, or Microsoft Edge v42 and all subsequent releases, providing users with flexibility in their choice of browser."},
    {"id": 16, "category": "Software", "description": "The software should include a repository or be compatible with Database Management Systems (DBMS) to store log data, alerts, and dashboards, ensuring efficient data management and accessibility."},
    {"id": 17, "category": "Software", "description": "The software should have the capacity to store a minimum of 20 TB of data in its repository, ensuring ample space for data accumulation over time."},
    {"id": 18, "category": "Software", "description": "The software should operate seamlessly on networks running IPv4, ensuring compatibility with widely used networking standards."},
    {"id": 19, "category": "Software", "description": "The software should operate seamlessly on networks running IPv6, ensuring forward compatibility with the latest networking standards."},
    {"id": 20, "category": "Software", "description": "The software should use TLS versions 1.2 or later for remote connections, ensuring secure, encrypted communication over networks."},
    {"id": 21, "category": "Software", "description": "The software should use both TLS versions 1.2 or later for remote connections and be configured to use FIPS-based cipher suites, providing a high level of security and data protection."},
    {"id": 22, "category": "Software", "description": "The software should use X.509 version 3 TLS certificates for mutual authentication, ensuring secure communication between the software and its users."},
    {"id": 23, "category": "Software", "description": "The software should allow CRA to integrate existing credential access management systems, providing seamless integration with existing infrastructure."},
    {"id": 24, "category": "Software", "description": "The software should use file integrity verification mechanisms to ensure the integrity of the logs, preventing unauthorized modification."},
    {"id": 25, "category": "Software", "description": "The software should include encryption capability for all data at rest using the FIPS 140-2 compliant AES algorithm, ensuring the security of stored data."},
    {"id": 26, "category": "Software", "description": "The software should use either RESTful or SOAP APIs, allowing it to integrate easily with a variety of other software and services."},
    {"id": 27, "category": "Software", "description": "The software should be deployable in both production and testing environments, providing flexibility in its deployment and allowing for thorough testing before production use."},
    {"id": 28, "category": "Software", "description": "The software should permit deployment for up to 10 instances of the software, ensuring scalability to meet growing demands."},
    {"id": 29, "category": "Software", "description": "The software should read log entries stored at different tiers based on age and criticality, ensuring efficient and intelligent data management."},
    {"id": 30, "category": "Software", "description": "The software should be capable of collecting a minimum of 12 GB of log data per hour, ensuring the ability to handle high volumes of data in a timely manner."},
    {"id": 31, "category": "Software", "description": "The software should collect logs from a minimum of 500 servers, ensuring scalability to meet the needs of large server clusters."},
    {"id": 32, "category": "Software", "description": "The software should be capable of directly ingesting and interpreting various log types from the Mainframe, ensuring wide compatibility with different log formats."},
    {"id": 33, "category": "Software", "description": "The software should be capable of directly ingesting and interpreting various log types from the server hosting the logs, ensuring wide compatibility with different log formats."},
    {"id": 34, "category": "Software", "description": "The software should accept as input and process log files up to and including 500GB in size, ensuring the ability to handle large amounts of data."},
]


default_file_path = 'https://rfp-cases.s3.ca-tor.cloud-object-storage.appdomain.cloud/2023/Bell/C09AD188-0000-C922-B40E-E6AA3F3B9A1F/906090ae-344d-4108-a5d8-9ca42f3fc292/906090ae-344d-4108-a5d8-9ca42f3fc292.pdf'

@app.get("/step2")
async def get_title_summary_date(file_path: str = default_file_path) -> RFPDetails:
    title = "Log Analytics Software"
    summary = "The Canada Revenue Agency (CRA) is seeking a Log Analytics Software to augment their Application Performance Monitoring (APM) tools. The aim is to enhance the proactive identification of performance issues before they escalate and cause service outages. The software will collect, aggregate, correlate, and analyze machine data from various systems across the CRA’s Linux network and mainframe to provide real-time insights into application performance."
    date = "June 7, 2021 at 2:00 P.M. EDT"
    methods_of_submission = "Proposals must be physically delivered to the Bid Receiving Unit at the Ottawa Technology Centre, 875 Heron Road, Room D-95, Ottawa, ON K1A 1A2. Electronic bids, including those sent by email or facsimile, will not be accepted."

    return RFPDetails(title=title, summary=summary, due_date=date, methods_of_submission=methods_of_submission)

@app.get("/step3_raw")
async def get_rfp_raw(file_path: str = default_file_path) -> RequirementDetails:
    return requirements

@app.get("/step3_classify")
async def get_rfp_classify(file_path: str = default_file_path) -> ClassifiedRequirements:
    response = ClassifiedRequirements(classified_requirements=classified_requirements_list)
    return response

@app.get("/step3_enrich")
async def get_rfp_enrich(file_path: str = default_file_path) -> RequirementDetails:
    result = EnrichmentRequirement(requirements=[Requirement(**req) for req in enriched_requirements])
    return result
