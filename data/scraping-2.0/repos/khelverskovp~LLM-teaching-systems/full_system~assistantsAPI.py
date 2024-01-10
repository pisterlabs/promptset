import os
import chromadb
import openai

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_transformers import (
    LongContextReorder,
)
from langchain.chains import StuffDocumentsChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from gptFunctions import * 

from openai import OpenAI
import json
import time
from utils import *

import aspose.words as aw


# Bad idea but works
#import ssl
#ssl._create_default_https_context = ssl._create_unverified_context


initial_loop = True

client = OpenAI(
    api_key='your-api-key-here',
)

# What subject is the document about
subject = input("What subject is the document about? \nAnswer: ")

# What is the document path
document_path = input("Enter the path to your educational material: ")

# Check if video
file_name = document_path

# Loop through all files in the current directory
if file_name.endswith(".mp4") or file_name.endswith(".wav") or file_name.endswith(".flac") or file_name.endswith(".m4a"):
    
    # make a sound file
    if file_name.endswith(".mp4") or file_name.endswith(".flac") or file_name.endswith(".m4a"):
        file_name_sound = makesound(file_name)
        print(file_name_sound)

    # transcribe using whisper
    print(f"Transcribing {file_name_sound}")
    transcribeTimestamp(file_name_sound)
    # make pdf 
    # load TXT document
    doc = aw.Document(file_name.split('.')[0]+'_transcript.txt')

    # save TXT as PDF file
    doc.save(file_name.split('.')[0]+'.pdf', aw.SaveFormat.PDF)

    # and set the document path to the pdf
    document_path = file_name.split('.')[0]+'.pdf'

print(f"Document path: {document_path}")

# Get pages from the PDF
pages = getPages(document_path)
print('Getting pages from PDF...')

# Set the type of embedding to use
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Make a list of the text from each page
texts = [page.page_content for page in pages]
text_ids = [str(page.metadata['page']) for page in pages]

# make dictionary of text and text_ids
text_dict = dict(zip(texts, text_ids))

# Create a retriever with some search kwargs (not refined, but works)
if len(pages) < 20: 
    k=2
else:
    k=10

retriever = Chroma.from_texts(texts, embedding=embeddings,ids=text_ids).as_retriever(
    search_kwargs={"k": k}
)

# Get the question
question = input("What is your question? \n Your answer/question: ")
docs = retriever.get_relevant_documents(question)

# get the page number of the most relevant page
relevant_page = int(text_dict[docs[0].page_content])

# print the most relevant page
#print(f'The most similar document to the query "{question}" is page {relevant_page} with content: \n\n "{docs[0].page_content}"')

# slice pdf using the page number
assistant_document = splitPDF(document_path,relevant_page)
print(f'Created a new PDF file with the most relevant page: {assistant_document}')


# We start by creating a file
file = client.files.create(
  file=open(assistant_document, "rb"),
  purpose='assistants'
)

# if the first document was a video, upload the whole .pdf file not just the RAG
print(f'Uploading the whole {document_path} to OpenAI API because it is a video...')
if file_name.endswith(".mp4") or file_name.endswith(".wav") or file_name.endswith(".flac") or file_name.endswith(".m4a"):
    file = client.files.create(
    file=open(document_path, "rb"),
    purpose='assistants'
    )

# Create an assistant with the subject
assistant = client.beta.assistants.create(
  name="Teacher Assistant",
  instructions=f'''You are a teacher in the {subject}. Given the context information and your prior knowledge, generate an appropriate guidance based on the question and the information from the source material.

    Your guidance should be step by step, such that you wait for a student to help answer its own question. In a way it should be socratic dialogue based. 

    Your answers should be guiding, trying to help the student learn. Don't give the answers away, guide the student iteratively. Simplify the prodecure step by step and wait for the students responses. 
                ''',
  model="gpt-4-1106-preview",
  tools=[{"type": "retrieval"}],
  file_ids=[file.id]
)

# We create a thread
thread = client.beta.threads.create()

# We get the thead id
main_thread = thread.id

# keep track of iterations
i = 0

# make a list of strings that contains the 

while True:
    # Ask the user what he/she needs help with
    if initial_loop:
        user_message = question
        initial_loop = False
        photo = False

    else:
            
        user_message = input("\nYour answer/question: ")

        # ask the user if she has a photo she wants to upload
        photo_answer = input("\nDo you have a photo you want to upload? (y/n) \nAnswer: ")
        
        if photo_answer.lower() == 'yes' or photo_answer.lower() == 'y':

            # photo yes/no boolean
            photo = True

            # ask the user to enter the path to the photo
            photo_path = input("\nPlease enter the path to your photo: ")

            # encode the image
            image = encode_image(photo_path)
        
        else: 
            photo = False


    print(f"\nWaiting for assistant to respond...")

    if photo: 
        # Pass the user message, with the last 3 thread, and the photo
        # into the getVisionResponse() function

        # Try getting the last 3 messages
        try:
            context_messages = [messages.data[i].content[0].text.value for i in range(0,6)]
        except:
            context_messages = [messages.data[i].content[0].text.value for i in range(0,2*i)]

        # reverse the context messages
        context_messages.reverse()

        # for every other message, add 'user' or 'assistant' to the message
        for i in range(len(context_messages)):
            if i%2 == 0:
                context_messages[i] = 'user: ' + context_messages[i]
            else:
                context_messages[i] = 'assistant: ' + context_messages[i]

        # joing the context messages into one string
        context_messages = '\n'.join(context_messages)

        # get the response from the API
        assistant_response = getVisionResponse(photo_path,context_messages+user_message)
        
        print(f"Waiting for GPT-4 Vision to respond...")
        print(f'\nGPT-4 Vision response: {assistant_response}')

        # add the user message to the thread
        message = client.beta.threads.messages.create(
            thread_id=main_thread,
            role="user",
            content= user_message + '\n You have the following knowledge of the a photo given from the user in relation to the conversation (as the assistant). Summarize the photo in your next response and use it in the conversation: \n' + assistant_response
        )


    else: 
        # We can add a message to the thread
        message = client.beta.threads.messages.create(
            thread_id=main_thread,
            role="user",
            content=user_message,
        )

    # We associate the assistant to the thread!
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
    )

    # Since runs is asynchronous, we need to wait for it to complete. 
    # We do this by querying the run until it is completed.
    def run_and_wait(run):
        start_time = time.time()
        timer = 0
        while run.status != "completed":
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id,
            )
            
            time.sleep(1/4)
            timer = time.time() - start_time

            print(f"\nWaiting for run to complete... {timer:.2f} seconds elapsed", end="\r", flush=True) 
        
        return run

    # We call the function to wait for the run to complete
    run = run_and_wait(run)

    # We can then print the response!
    messages = client.beta.threads.messages.list(thread_id=thread.id)


    # save the thread id in a list that lists thread ids in order
    print(messages.data[0].content[0].text.value)

    i += 1