# framework imports
from fastapi import APIRouter, status, HTTPException, File, UploadFile
import os
import tempfile
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import BSHTMLLoader
from openai import OpenAI
from fastapi import Form
import asyncio

from src.services.embeddings_manager import setup_embedding_model
from src.services.milvus_manager import setup_milvus
from src.services.embeddings_processor import EmbeddingsProcessor
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel
from typing import Any

class AbstractModel(BaseModel):
    """Schema Models

    Args:
        BaseModel (_type_): Inherits from Pydantic and specifies Config
    """

    class Config:
        orm_mode = True
        use_enum_values = True


def scrape_html_from_url(url):
    try:
        # Send an HTTP GET request to the URL
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        response = requests.get(url, headers=headers, verify=False)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the HTML content using BeautifulSoup
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract the text content from the HTML
            html_content = soup.get_text()
            # Create a temporary file to save the HTML content
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".html", encoding="utf-8"
            ) as html_file:
                html_file.write(html_content)

            html_path = html_file.name
            print(f"HTML content saved to {html_path}")
            return html_path

        else:
            print(f"Failed to fetch HTML content. Status code: {response.status_code}")
            return None

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None


milvus_manager = None
embeddings_model = None
embeddings_processor = None

if "MILVUS_URI" in os.environ:
    milvus_manager = setup_milvus()
    embeddings_model = setup_embedding_model()
    embeddings_processor = EmbeddingsProcessor(embeddings_model, milvus_manager)



# API Router
core_router = APIRouter(prefix="/api/v1/core", tags=["Core APIs"])


class HotelData(AbstractModel):
    agentName: str
    agentRole: str
    hotelName: str
    hotel: str


@core_router.post("/add_hotel/", status_code=status.HTTP_201_CREATED)
async def add_hotel(
    hotelName: str = Form(...),
    hotelURL: str = Form(...),
    agentRole: str = Form(...),
    agentName: str = Form(...),
    upload_document: UploadFile = File(...),
):
    try:
        # Verify if the uploaded document is a PDF
        if not upload_document.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Only PDF files are allowed")

        # Create a temporary directory to store the uploaded PDF
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_path = os.path.join(
                "/home/azam/projects/inhotel/uploads", upload_document.filename
            )
            # Save the uploaded PDF to the temporary directory
            with open(pdf_path, "wb") as pdf_file:
                pdf_file.write(upload_document.file.read())

            # Check if the file was successfully created
            if not os.path.exists(pdf_path):
                raise HTTPException(status_code=500, detail="Failed to create the file")

        # Parsing and chunking the document.
        print("Inserting document data into Milvus")
        doc_data = PyPDFLoader(pdf_path).load_and_split(
            RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        )
        # Embedding and insert chunks into the vector database.

        for doc in doc_data:
            embeddings_processor.process_and_save("".join([doc.page_content]))
        print("Done Insertion")

        print("Inserting web data into Milvus")
        if hotelURL:
            web_data = BSHTMLLoader(scrape_html_from_url(hotelURL)).load_and_split(
                RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            )
            for doc in web_data:
                embeddings_processor.process_and_save("".join([doc.page_content]))
        print("Done Insertion")

        # Setup OpenAI client.
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        # Set default values for optional fields
        hotelName = hotelName or "Hotel"
        agentRole = agentRole or "Front Desk Officer"
        agentName = agentName or "Emily"

        # Create an Assistant.
        my_assistant = client.beta.assistants.create(
            name=f"Chat with {agentName}",
            instructions=f"You are {agentName}, a {agentRole} at a renowned hotel named {hotelName}. You excel at assisting others by answering their queries and providing relevant information. You can search for pertinent information using Hotel Database tool if required and respond to questions based on the information retrieved. When you are unsure of an answer to a question, you admit your lack of knowledge while ensuring you always remain polite.",
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "CustomRetriever",
                        "description": "Retrieve relevant information of Chedi Penthouse",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The user query",
                                }
                            },
                            "required": ["query"],
                        },
                    },
                }
            ],
            model="gpt-4-1106-preview",
        )

        # Return status and assistant ID
        return {
            "status": "Data processed successfully",
            "assistant_id": my_assistant.id,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing data: {str(e)}")

class WebhookPayload(BaseModel):
    event: str
    payload: Any



@core_router.post('/api/webhook', status_code=status.HTTP_201_CREATED)
async def webhook(data: WebhookPayload):
    if data.event == 'webhook:verify':
        # Respond with the random string in the payload
        return data.payload
    elif data.event == 'message:created':

        body = data.payload.get("body")
        if len(body) > 30 :
            return {'ok': True}
        conversation_id = data.payload.get("conversation_id")

        assistant_ID="asst_mtV7ZBVjWmjLBf7DsEbE1WK5"
        user_message=body
        conversation_id=conversation_id
        thread_id=None

        # Directly call chat_hotel function
        chat_response = await chat_hotel(
            assistant_ID=assistant_ID,
            thread_id=thread_id,
            message=user_message
        )

        API_KEY="SFMyNTY.g2gDbAAAAAJoAmQAB3VzZXJfaWRiAAAfaGgCZAAKYWNjb3VudF9pZG0AAAAkZjYzMDc4ZjgtYTQ0NS00OTFlLWEzYTMtZjFjMzkwMGI5NTkyam4GAHxs8cOMAWIAAVGA.W0UOWmYiPppwStrK6m6hlmIkeD0EhBxeGSHEvqAPZsI"
        headers = {
            'Authorization': f'Bearer {API_KEY}',
            'Content-Type': 'application/json'
        }
        payload = {
            'message': {
                'body': chat_response.get("message"),
                'conversation_id': conversation_id
            }
        }

        # Send a POST request to the Papercups API
        response = requests.post('https://app.papercups.io/api/v1/messages', headers=headers, json=payload)


        # Handle the response from chat_hotel
        if  chat_response.get("message"):
            print("Chat response:",  chat_response.get("message"))
        else:
            print("No response from chat_hotel")

        return {'ok': True}

@core_router.post("/chat_hotel/", status_code=status.HTTP_201_CREATED)
async def chat_hotel(
    assistant_ID: str = Form(None),
    thread_id: str = Form(None),
    message: str = Form(None),
):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OPENAI_API_KEY is not set in the environment variables",
        )

    client = OpenAI(api_key=api_key)

    try:
        # Initialize or update the thread with the user message
        if not thread_id:
            thread = client.beta.threads.create(
                messages=[{"role": "user", "content": message}]
            )
            thread_id = thread.id
        else:
            client.beta.threads.messages.create(
                thread_id=thread_id, role="user", content=message
            )

        # Create a new run
        run = client.beta.threads.runs.create(
            thread_id=thread_id, assistant_id=assistant_ID
        )
        max_retries = 35
        while max_retries > 0:
            await asyncio.sleep(1)
            run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
            if run.status == "requires_action":
                tool_outputs = await process_tool_calls(run, message, client)
                run = client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread_id, run_id=run.id, tool_outputs=tool_outputs
                )
            elif run.status not in ["queued", "in_progress"]:
                break

            max_retries -= 1

        if max_retries == 0:
            raise TimeoutError("Run did not complete in time")

        # Retrieve and format the final conversation messages
        messages = client.beta.threads.messages.list(thread_id=thread_id)
        for m in messages:
            if m.role == "assistant":
                return {"message": m.content[0].text.value, "thread_id": thread_id}

    except HTTPException as http_ex:
        raise http_ex
    except Exception as ex:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing data: {str(ex)}",
        )


async def process_tool_calls(run, user_message, client):
    tool_outputs = []
    for tool_call in run.required_action.submit_tool_outputs.tool_calls:
        if tool_call.function.name == "CustomRetriever":
            hits = embeddings_processor.process_and_search(user_message, top_k=3)
            hit_titles = [
                document.entity.get("text")
                for hit in hits["hits_data"]
                for document in hit
            ]
            tool_outputs.append(
                {"tool_call_id": tool_call.id, "output": ("\n\n").join(hit_titles)}
            )
    return tool_outputs
