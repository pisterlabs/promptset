
from django.shortcuts import render
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from .forms import RegistrationForm, PDFUploadForm
from django.contrib.auth import login, authenticate, logout
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.models import User
from .models import ChatMessage, PDFDocument
from langchain.chat_models import ChatOpenAI
from django.contrib.auth.decorators import login_required
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from django.http import HttpResponse, JsonResponse
import fitz  # PyMuPDF
from django.http import JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.views import View
from django.shortcuts import render, get_object_or_404, redirect
from .models import PDFDocument
from .forms import PDFUpdateForm, PDFDocumentForm2
import os
from django.http import HttpResponseNotFound
from dotenv import load_dotenv


vector_store = None
conversation_chain = None 

pdfname=None
pdfsize=None
scripttext=None

def get_vectorstore(text_chunks):
    """
    Retrieves the vector store for text chunks.

    :param text_chunks: List of text chunks.
    :return: Knowledge base vector store.
    """
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(text_chunks, embeddings)
    return knowledge_base

def get_conversation_chain(vectorstore):
    """
    Retrieves the conversation chain.

    :param vectorstore: Vector store.
    :return: Conversational retrieval chain.
    """
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain





def get_pdf_text(pdf):
    """
    Retrieves text from a PDF document.

    :param pdf: PDF file.
    :return: Extracted text from the PDF.
    """
    if pdf:
        pdf_reader = PdfReader(pdf)
        text = ''.join(page.extract_text() for page in pdf_reader.pages)
    return text

def get_text_chunks(text):
    """
    Splits text into chunks.

    :param text: Input text.
    :return: List of text chunks.
    """
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text)
    # print(chunks)
    return chunks

def get_conversation_chain(vectorstore):
    
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

  

@login_required(login_url="/login/")
def upload_pdf(request):
    """
    Handles the PDF upload.

    :param request: HTTP request.
    :return: JSON response.
    """
    if request.method == 'POST':
        form = PDFUploadForm(request.POST, request.FILES)
        if form.is_valid():
            pdf_document = request.FILES['pdf_document']
            # Save the PDF document to the database
            pdf = PDFDocument(user=request.user, title=pdf_document.name)
            pdf.documentContent = process_uploaded_pdf(pdf_document)
            pdf.save()
            return JsonResponse({'message': 'PDF uploaded successfully.'}, status=200)
        else:
            return JsonResponse({'error': 'Invalid form data.'}, status=400)
    else:
        form = PDFUploadForm()
    return render(request, 'ask_question.html', {'form': form})

def process_uploaded_pdf(pdf_file):
    """
    Processes the uploaded PDF document.

    :param pdf_file: Uploaded PDF file.
    :return: Extracted raw text from the PDF.
    """
    raw_text = get_pdf_text(pdf_file)
    # print("Extracted PDF text:", raw_text)
    # text_chunks = get_text_chunks(raw_text)
    # print(text_chunks)
    return raw_text



@login_required(login_url="/login/")
def ask_question(request):
    """
    Handles the user's question and generates a response.

    :param request: HTTP request.
    :return: Rendered page with the response.
    """
    load_dotenv()
    chat_history = ChatMessage.objects.filter(user=request.user).order_by('timestamp')  # Retrieve chat history for the logged-in user
    chat_response = ''
    user_pdfs = PDFDocument.objects.filter(user=request.user)
    user_question = ""

    if request.method == 'POST':
        user_question = request.POST.get('user_question')
        selected_pdf_id = request.POST.get('selected_pdf')
        selected_pdf = get_object_or_404(PDFDocument, id=selected_pdf_id)
        text_chunks = get_text_chunks(selected_pdf.documentContent)

        knowledge_base = get_vectorstore(text_chunks)

        docs = knowledge_base.similarity_search(user_question)

        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
            response = chain.run(input_documents=docs, question=user_question)

        chat_response = response
        print(chat_response)
        chat_message = ChatMessage(user=request.user, message=user_question, answer=chat_response)
        chat_message.save()

    context = {'chat_response': chat_response, 'chat_history': chat_history, 'user_question': user_question}

    return render(request, 'ask_question.html', {'user_pdfs': user_pdfs, **context})



def process_user_question(pdf, user_question):
    """
    Processes the user's question with a PDF document.

    :param pdf: PDF document content.
    :param user_question: User's question.
    :return: Response to the user's question.
    """
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(pdf, embeddings)
    
    docs = knowledge_base.similarity_search(user_question)
    
    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=user_question)
        
        return response

    
@login_required(login_url="/login/")
def view_pdf(request, pdf_id):
    """
    Displays a PDF document.

    :param request: HTTP request.
    :param pdf_id: ID of the PDF document.
    :return: Rendered page with the PDF document.
    """
    pdf = PDFDocument.objects.get(id=pdf_id)
    return render(request, 'view_pdf.html', {'pdf': pdf})

@login_required(login_url="/login/")
def view_chat_history(request):
    """
    Displays the chat history for the logged-in user.

    :param request: HTTP request.
    :return: Rendered page with chat history.
    """
    chat_messages = ChatMessage.objects.filter(user=request.user)
    return render(request, 'view_chat_history.html', {'chat_messages': chat_messages})




def list_pdfs(request):
    """
    Lists PDF documents for the logged-in user.

    :param request: HTTP request.
    :return: Rendered page with the list of PDF documents.
    """
    pdfs = PDFDocument.objects.filter(user=request.user)
    return render(request, 'edit_pdf.html', {'pdfs': pdfs})


def delete_pdf(request, pdf_id):
    """
    Deletes a PDF document.

    :param request: HTTP request.
    :param pdf_id: ID of the PDF document to delete.
    :return: Redirect or error response.
    """
    try:
        pdfs = PDFDocument.objects.get(id=pdf_id)
        pdf_title = getattr(pdfs, 'document', pdfs.title)
        pdfs.delete()
        return redirect('/pdfs/')
    except PDFDocument.DoesNotExist:
        # Handle the case where the PDFDocument with the given id does not exist
        return HttpResponseNotFound("PDF not found")


def update_pdf(request, pdf_id):
    """
    Updates a PDF document.

    :param request: HTTP request.
    :param pdf_id: ID of the PDF document to update.
    :return: Rendered page with the updated PDF document.
    """
    pdf = get_object_or_404(PDFDocument, pk=pdf_id)
    if request.method == 'POST':
        form = PDFDocumentForm2(request.POST, instance=pdf)
        if form.is_valid():
            form.save()
            messages.success(request, 'Pdf updated successfully.')
            return redirect('/pdfs/')
    else:
        form = PDFUpdateForm(instance=pdf)
    return render(request, 'update_pdf.html', {'form': form, 'pdf': pdf})















































  
    
# def upload_pdf(request):
#     if request.method == 'POST':
#         form = PDFUploadForm(request.POST, request.FILES)
#         if form.is_valid():
#             pdf_document = request.FILES['pdf_document']
           
#             # Save the PDF document to the database and process it here
#             pdf = PDFDocument(user=request.user, title=pdf_document.name)
#             pdf.save()
#             pdf.documentContent = process_uploaded_pdf(pdf_document)
#             pdf.save()
#             return JsonResponse({'message': 'PDF uploaded successfully.'}, status=200)
#         else:
#             return JsonResponse({'error': 'Invalid form data.'}, status=400)
#     else:
#         form = PDFUploadForm()
#     return render(request, 'ask_question.html', {'form': form})



# def ask_question(request):
    chat_history = ChatMessage.objects.filter(user=request.user).order_by('timestamp')  # Retrieve chat history for the logged-in user
    chat_response= ''
    user_pdfs = PDFDocument.objects.filter(user=request.user)
    if request.method == 'POST':
        user_question = request.POST.get('user_question')
        selected_pdf_id = request.POST.get('selected_pdf')
        selected_pdf = get_object_or_404(PDFDocument, id=selected_pdf_id)
        text_chunks = get_text_chunks(selected_pdf.documentContent)
        print(text_chunks)
        
        knowledge_base = get_vectorstore(text_chunks)
        
        docs = knowledge_base.similarity_search(user_question)
        
        llm = OpenAI()
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
        #         # print(response)
        
        chat_response = response
        print(chat_response)
        chat_message = ChatMessage(user=request.user, message=user_question, answer=chat_response)
        chat_message.save()
        
        contexts = {'chat_response': chat_response, 'chat_history': chat_history}
        
        context = {'chat_response': chat_response, 'chat_history': chat_history, 'user_question': user_question}

        
    return render(request, 'ask_question.html', {'user_pdfs': user_pdfs, 'context': context})
