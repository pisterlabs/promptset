import os
import json
from dotenv import main
from datetime import datetime
from openai import OpenAI
from fastapi import FastAPI, Request, HTTPException, status, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi.security import APIKeyHeader
from nostril import nonsense
import re
import time
import asyncio
import boto3
from botocore.exceptions import NoCredentialsError


# Initialize environment variables
main.load_dotenv()

# Initialize backend & API keys
server_api_key=os.environ['BACKEND_API_KEY'] 
API_KEY_NAME=os.environ['API_KEY_NAME'] 
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key_header: str = Depends(api_key_header)):
    if not api_key_header or api_key_header.split(' ')[1] != server_api_key:
        raise HTTPException(status_code=401, detail="Could not validate credentials")
    return api_key_header

# Initialize the SQS client
sqs_client = boto3.client('sqs', region_name='your-region')

# Function to send message to SQS
def send_message_to_sqs(queue_url, message_body):
    try:
        response = sqs_client.send_message(
            QueueUrl=queue_url,
            MessageBody=message_body
        )
        return response
    except NoCredentialsError:
        print("Credentials not available")
        return None

# Define query class
class Query(BaseModel):
    user_input: str
    user_id: str
    user_locale: str | None = None

# Define FastAPI app
app = FastAPI()

# Initialize OpenAI client & Embedding model
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

# Initialize email address detector
email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
def find_emails(text):  
    return re.findall(email_pattern, text)

# Set up address filters:
EVM_ADDRESS_PATTERN = r'\b0x[a-fA-F0-9]{40}\b|\b0x[a-fA-F0-9]{64}\b'
BITCOIN_ADDRESS_PATTERN = r'\b(1|3)[1-9A-HJ-NP-Za-km-z]{25,34}\b|bc1[a-zA-Z0-9]{25,90}\b'
LITECOIN_ADDRESS_PATTERN = r'\b(L|M)[a-km-zA-HJ-NP-Z1-9]{26,34}\b'
DOGECOIN_ADDRESS_PATTERN = r'\bD{1}[5-9A-HJ-NP-U]{1}[1-9A-HJ-NP-Za-km-z]{32}\b'
XRP_ADDRESS_PATTERN = r'\br[a-zA-Z0-9]{24,34}\b'
COSMOS_ADDRESS_PATTERN = r'\bcosmos[0-9a-z]{38,45}\b'
SOLANA_ADDRESS_PATTERN= r'\b[1-9A-HJ-NP-Za-km-z]{32,44}\b'
CARDANO_ADDRESS_PATTERN = r'\baddr1[0-9a-z]{58}\b'

# Initialize user state and periodic cleanup function
user_states = {}
TIMEOUT_SECONDS = 600  # 10 minutes

async def periodic_cleanup():
    while True:
        await cleanup_expired_states()
        await asyncio.sleep(TIMEOUT_SECONDS)

# Improved startup event to use asyncio.create_task for the continuous background task
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(periodic_cleanup())

# Enhanced cleanup function with improved error logging
async def cleanup_expired_states():
    try:
        current_time = time.time()
        expired_users = [
            user_id for user_id, state in user_states.items()
            if current_time - state['timestamp'] > TIMEOUT_SECONDS
        ]
        for user_id in expired_users:
            try:
                del user_states[user_id]
                print("User state deleted!")
            except Exception as e:
                print(f"Error during cleanup for user {user_id}: {e}")
    except Exception as e:
        print(f"General error during cleanup: {e}")

# Define exception handler function
@app.exception_handler(Exception)
async def generic_exception_handler(request, exc):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"message": "Snap! Something went wrong, please try again!"},
    )

# Define supported locales for data retrieval
SUPPORTED_LOCALES = {'eng', 'fr', 'ru'}

# Load classifier system prompt
def load_categories():
    filecat = f'classifier_prompt.txt'
    try:
        with open(filecat, 'r') as categories:
            return categories.read()
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"Categories not found!")

# Pre-load prompts
classifier_prompt = load_categories()

# Define helpers functions
def handle_nonsense(locale):
    messages = {
        'fr': "Je suis désolé, je n'ai pas compris votre question et je ne peux pas aider avec des questions qui incluent des adresses de cryptomonnaie. Pourriez-vous s'il vous plaît fournir plus de détails ou reformuler sans l'adresse ? N'oubliez pas, je suis ici pour aider avec toute demande liée à Ledger.",
        'ru': "Извините, я не могу понять ваш вопрос, и я не могу помочь с вопросами, содержащими адреса криптовалют. Не могли бы вы предоставить более подробную информацию или перефразировать вопрос без упоминания адреса? Помните, что я готов помочь с любыми вопросами, связанными с Ledger.",
        'default': "I'm sorry, I didn't quite get your question, and I can't assist with questions that include cryptocurrency addresses or transaction hashes. Could you please provide more details or rephrase it without the address? Remember, I'm here to help with any Ledger-related inquiries."
    }
    print('Nonsense detected!')
    return {'output': messages.get(locale, messages['default'])}

def handle_crypto_email(locale, context):
    context_dict = {
        'crypto': {
            'fr': "Je suis désolé, mais je ne peux pas aider avec des questions qui incluent des adresses de cryptomonnaie. Veuillez retirer l'adresse et poser la question à nouveau.",
            'ru': "Извините, но я не могу помочь с вопросами, которые включают адреса счетов криптовалюты. Пожалуйста, удалите адрес из вашего запроса и напишите ваш запрос еще раз.",
            'default': "I'm sorry, but I can't assist with questions that include cryptocurrency addresses. Please remove the address and ask again"
        },
        'email': {
            'fr': "Je suis désolé, mais je ne peux pas aider avec des questions qui incluent des adresses e-mail. Veuillez retirer l'adresse et poser la question à nouveau.",
            'ru': "Извините, но я не могу ответить на вопросы, содержащие адреса электронной почты. Пожалуйста, удалите адрес электронной почты и задайте вопрос снова.",
            'default': "I'm sorry, but I can't assist with questions that include email addresses. Please remove the address and ask again."
        }
    }
    print('Email or crypto address detected!')
    return {'output': context_dict[context].get(locale, context_dict[context]['default'])}

# Set server response dictionary
server_responses = {
    "greetings": {
        "fr": "Bonjour ! Comment puis-je vous aider avec vos problèmes liés à Ledger aujourd'hui ? Plus vous partagerez de détails sur votre problème, mieux je pourrai vous assister. ",
        "ru": "Здравствуйте! Как я могу помочь вам с вашими вопросами, связанными с Ledger, сегодня? Чем больше деталей вы предоставите о вашей проблеме, тем лучше я смогу вам помочь. Пожалуйста, опишите её максимально подробно!",
        "eng": "Hello! How can I assist you with your Ledger-related issue today? The more details you share about the problem, the better I can assist you. Feel free to describe it in as much detail as possible!"
    }
}

# Set patterns dictionary
patterns = {
    'crypto': [EVM_ADDRESS_PATTERN, BITCOIN_ADDRESS_PATTERN, LITECOIN_ADDRESS_PATTERN, 
            DOGECOIN_ADDRESS_PATTERN, COSMOS_ADDRESS_PATTERN, CARDANO_ADDRESS_PATTERN, 
            SOLANA_ADDRESS_PATTERN, XRP_ADDRESS_PATTERN],
    'email': [email_pattern]
}


######## ROUTES ##########

# Home route
@app.get("/")
async def root():
    return {"welcome": "You've reached the home route!"}

# Health probe
@app.get("/_health")
async def health_check():
    return {"status": "OK"}

# Categorizer route
@app.post('/categorizer')
async def react_description(query: Query, request: Request, api_key: str = Depends(get_api_key)): 
    user_id = query.user_id
    user_input = query.user_input.strip()
    locale = query.user_locale if query.user_locale in SUPPORTED_LOCALES else "eng"

    # Create a conversation history for new users
    timestamp = datetime.now().strftime("%B %d, %Y %H:%M:%S")
    user_states.setdefault(user_id, {
        'previous_queries': [],
        'timestamp': [],
        'category': [],
    })

    # Apply nonsense filter
    if not user_input or nonsense(user_input):
        return handle_nonsense(locale)
    
    # Apply email & crypto addresses filter
    for context, pattern_list in patterns.items():
        if any(re.search(pattern, user_input, re.IGNORECASE) for pattern in pattern_list):
            return handle_crypto_email(locale, context)

    else:
        
        try:
             
            # Categorize query using finetuned GPT model
            resp = client.chat.completions.create(
                    temperature=0.0,
                    model='ft:gpt-3.5-turbo-0613:ledger::8cZVgY5Q',
                    seed=0,
                    messages=[
                        {"role": "system", "content": classifier_prompt},
                        {"role": "user", "content": user_input}
                    ],
                    timeout=5.0,
                    max_tokens=50,
                )
            category = resp.choices[0].message.content.lower()
            print("Category: " + category)

            # Filter unwanted categories
            if category and category in server_responses:
                return {"output": server_responses[category].get(locale, server_responses[category]["eng"])}
            
            # Save the response to a thread
            user_states[user_id] = {
                'previous_queries': user_states[user_id].get('previous_queries', []) + [(user_input)],
                'timestamp': user_states[user_id].get('timestamp', []) + [(timestamp)],
                'category': user_states[user_id].get('category', []) + [(category)],
            }
            print(user_states)

            # Format .json object
            output_data = {
                "category": category,  
                "time": timestamp   
            }
            print(output_data)

            # Convert the output data to a string or a serialized format like JSON
            output_json = json.dumps(output_data)

            # # Send the message to SQS
            # sqs_queue_url = 'your-sqs-queue-url'
            # send_message_response = send_message_to_sqs(sqs_queue_url, output_json)
            # print(send_message_response)
                    
            return output_data

        except ValueError as e:
            # Log the error for debugging purposes
            print(f"ValueError occurred: {e}")
            return JSONResponse(
                status_code=400,
                content={"message": "Invalid input or configuration error. Please check your request."}
            )
        except HTTPException as http_exc:
            # Log the error for debugging purposes
            print(f"HTTPException occurred: {http_exc.detail}")
            return JSONResponse(
                status_code=http_exc.status_code,
                content={"message": http_exc.detail}
            )
        except Exception as e:
            # Log the error for debugging purposes
            print(f"Unexpected error occurred: {e}")
            return JSONResponse(
                status_code=500,
                content={"message": "An unexpected error occurred. Please try again later."}
            )

# Local start command: uvicorn categorizer:app --reload --port 8800