from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from decouple import config
from starlette.responses import JSONResponse
import requests
import base64
from openai import AzureOpenAI
from ..views.automate import AutoHackPromptBuilder

API_KEY = config('AZURE_OPENAI_API_KEY')
ENDPOINT = config('AZURE_OPENAI_ENDPOINT')
LLM = config('DEPLOYMENT_NAME_LLM')

azure_client = AzureOpenAI(
    api_key=API_KEY,
    api_version="2023-07-01-preview",
    azure_endpoint=ENDPOINT
)


chat_router = APIRouter()
vision_router = APIRouter()

@chat_router.post("/chat_router", response_model=dict)
async def chat(question: str):
    """
    Asynchronous endpoint to handle chat requests using an LLM.

    This endpoint takes a question as input and uses the AutoHackPromptBuilder
    to generate a system prompt which, along with the user's question, is sent
    to the LLM for processing. The LLM generates a completion based on the 
    provided messages and returns the response.

    Parameters
    ----------
    question : str
        The user's input question to be sent to the chat model.

    Returns
    -------
    dict
        A dictionary with a single key 'result' containing the LLM's response.

    Raises
    ------
    HTTPException
        An error response with status code 400 if an exception occurs during 
        the chat process.
    """
    try:
        autohack_prompt_builder = AutoHackPromptBuilder("BorusanAutoHack")
        sys_prompt_template = autohack_prompt_builder.generate_system_prompt(question)
        messages = [{"role": "system", "content": sys_prompt_template}, 
                    {"role": "user", "content": question}]

        completion = azure_client.chat.completions.create(
            model=LLM, 
            messages=messages
        )
        content = completion.choices[0].message.content

        if '#' in content:
            result_part = content.split('#')[0]
            type_part = content.split('#')[-1]
            if type_part in 'form-1':
                type_part = 'form-1'
            elif type_part in 'form-2':
                type_part = 'form-2'
            else:
                type_part = 'form-2'
        else:
            result_part = content
            type_part = None
        return {"result": result_part,
                "type": type_part}
    except Exception as ex:
        raise HTTPException(status_code=400, detail=f"Error in chat: {ex}")


@vision_router.post("/vision_router", response_model=dict)
async def vision(question: str = Form(...), file: UploadFile = File(...)):
    """
    Analyzes an uploaded image with a text query using GPT-4 Vision API.

    This endpoint encodes an uploaded image into base64 format and sends it
    with the text query to the GPT-4 Vision API. Returns the processed response focusing on
    the AI's interpretation of the image and the related question.

    Parameters
    ----------
    question : str
        Description or question about the image to be analyzed.
    file : UploadFile
        The image file to be analyzed, uploaded by the user.

    Returns
    -------
    dict
        Extracted content from the API response, containing the AI's interpretation of the image.

    Raises
    ------
    HTTPException
        In case of any error during the request to the API.

    Example
    -------
    >>> response = await vision("Describe this photo.", "http://example.com/photo.jpg")
    >>> print(response)
    """
    try:
        api_key = config('GPT4V_API_KEY')
        gpt4v_endpoint = config('GPT4V_ENDPOINT')

        # Dosyayı oku ve base64'e çevir
        file_contents = await file.read()
        encoded_image = base64.b64encode(file_contents).decode('ascii')

        headers = {
            "Content-Type": "application/json",
            "api-key": api_key,
        }

        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are an AI assistant that helps people find information."
                        }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded_image}"
                            }
                        },
                        {
                            "type": "text",
                            "text": question
                        }
                    ]
                }
            ],
            "temperature": 0.3,
            "top_p": 0.95,
            "max_tokens": 800
        }

        response = requests.post(gpt4v_endpoint, headers=headers, json=payload)
        response.raise_for_status()
        response_data = response.json()
        return JSONResponse(content={"result": response_data["choices"][0]["message"]["content"]})
    except Exception as ex:
        raise HTTPException(status_code=400, detail=f"Error processing text: {ex}")
