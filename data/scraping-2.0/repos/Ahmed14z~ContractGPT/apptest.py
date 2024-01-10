from flask import Flask, request, jsonify
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import service_pb2_grpc
from langchaincoexpert.llms import Clarifai
from langchaincoexpert.agents import load_tools

# import csv
import spacy
from pprint import pprint
from fpdf import FPDF
from langchaincoexpert.agents import initialize_agent
from langchaincoexpert.utilities import GoogleSearchAPIWrapper# import csv
from langchaincoexpert.agents import AgentType

from dropbox_sign import \
    ApiClient, ApiException, Configuration, apis, models

from langchaincoexpert.memory import VectorStoreRetrieverMemory
from langchaincoexpert.chains import ConversationChain
from langchaincoexpert.prompts import PromptTemplate
from langchaincoexpert.vectorstores import SupabaseVectorStore
from langchaincoexpert.embeddings import ClarifaiEmbeddings
from supabase.client import  create_client
# from dotenv import load_dotenv
# from firestore import db
from firebase_admin import firestore
# from dotenv import load_dotenv
import os
from flask_cors import CORS

# Load environment variables from .env file
# load_dotenv()
app = Flask(__name__)

CORS(app, supports_credentials=True)

# Clarifai settings
CLARIFAI_PAT = os.getenv("CLARIFAI_PAT")

# Supabase settings
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# os.environ['SERPER_API_KEY'] = "85a3148783d1539b6f59b5eca1968edd4d66f0d1"
tools = load_tools(["google-serper"])


nlp = spacy.load("en_core_web_sm")

# Set up the Clarifai channel
channel = ClarifaiChannel.get_grpc_channel()
stub = service_pb2_grpc.V2Stub(channel)

# Clarifai settings
USER_ID = 'ahmedz'
APP_ID = 'FINGU'
MODEL_ID = 'GPT-3_5-turbo'

#Drop Box Config
configuration = Configuration(
    # Configure HTTP basic authorization: api_key
    username="afcd15c5bb48d8034a8b8c9cad85978200b25f173f4697adce2768faa13b91d9",

    # or, configure Bearer (JWT) authorization: oauth2
    # access_token="YOUR_ACCESS_TOKEN",
)


# Initialize Clarifai embeddings
embeddings = ClarifaiEmbeddings(pat=CLARIFAI_PAT, user_id="openai", app_id="embed", model_id="text-embedding-ada")

# Initialize Supabase vector store
# vectordb = SupabaseVectorStore.from_documents({}, embeddings, client=supabase)

# Initialize Clarifai LLM
llm = Clarifai(pat=CLARIFAI_PAT, user_id='openai', app_id='chat-completion', model_id='GPT-4')


# Handle incoming messages
def handle_message(input_text , user_id,internet,spell,assesment):
    memory_key = {user_id}
    if internet : 

        response =  generate_Internet_response_llmchain(input_text, user_id)
    else: 
        response = generate_response_llmchain(input_text, user_id,spell=spell,assessment=assesment)

    
    return response


def generate_Internet_response_llmchain(prompt, conv_id):
    convid = "a" + str(conv_id)
    vectordb = SupabaseVectorStore.from_documents({}, embeddings, client=supabase,user_id=conv_id) # here we use normal userid "for saving memory"

    retriever = vectordb.as_retriever(search_kwargs=dict(k=15,user_id=convid)) # here we use userid with "a" for retreiving memory
    memory= VectorStoreRetrieverMemory(retriever=retriever , memory_key="chat_history")
    DEFAULT_TEMPLATE = """he following is a friendly conversation between a human and an AI called ContractGPT. 
   ,The Ai is a Contract Creation assitant designed to make Contracts.
   If the AI does not know the answer to a question, it truthfully says it does not know or reply with the same question.
   The AI should usually reply with the contract only without any instructions or explainations.
   
{history}
(You do not need to use these pieces of information if not relevant)

Current conversation:
Human: {input}
AI:"""

  
    agent = initialize_agent(tools, llm, agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory = memory)
    # agent.input_keys= 
    final = agent.run(input = prompt)
    return final


def generate_response_llmchain(prompt, conv_id,spell,assessment):
    convid = "a" + str(conv_id)
    # filter = {"user_id": userid}
    vectordb = SupabaseVectorStore.from_documents({}, embeddings, client=supabase,user_id=conv_id) # here we use normal userid "for saving memory"

    retriever = vectordb.as_retriever(search_kwargs=dict(k=10,user_id=convid)) # here we use userid with "a" for retreiving memory
    memory = VectorStoreRetrieverMemory(retriever=retriever, memory_key=convid)

    if spell:
        DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and an AI called ContractGPT. 
   ,The Ai is a Contract Creation assitant designed to make Contracts.
    The AI Should only check any spelling mistakes and grammer mistakes and return the contract with the spelling and grammer mistakes fixed while making the difference in bold.

Relevant pieces of previous conversation:
{user_id}
(You do not need to use these pieces of information if not relevant)

Current conversation:
Human: {input}
AI:"""
    if assessment:
           DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and an AI called ContractGPT. 
   ,The Ai is a Contract Creation assitant designed to make Contracts.
    The AI Should only make an overall risk assesment to the contract and give notes and advices.
Relevant pieces of previous conversation:
{user_id}
(You do not need to use these pieces of information if not relevant)

Current conversation:
Human: {input}
AI:"""
    else:
        DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and an AI called ContractGPT. 
   ,The Ai is a Contract Creation assitant designed to make Contracts.
   If the AI does not know the answer to a question, it truthfully says it does not know or reply with the same question.
   The AI should act as a tool that outputs a contract , and only asks questions when needed too. 
   

Relevant pieces of previous conversation:
{user_id}
(You do not need to use these pieces of information if not relevant)

Current conversation:
Human: {input}
AI:"""
    
    formatted_template = DEFAULT_TEMPLATE.format(user_id="{"+convid+"}",input = "{input}")

    PROMPT = PromptTemplate(
        input_variables=[convid, "input"], template=formatted_template
    )

    
    conversation_with_summary = ConversationChain(
        llm=llm,
        prompt=PROMPT,
        memory=memory,
        verbose=True
    )

    ans = conversation_with_summary.predict(input=prompt)
    # response = ans
    return ans

def text_to_pdf(text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(190, 10, txt=text, align="L")
    pdf_file_path = "output.pdf"
    pdf.output(pdf_file_path)
    return pdf_file_path

@app.route('/delete', methods=['DELETE'])
def deleteChat():
    data = request.get_json()
    conversationId = "a" + data['convid']
    
    # Check if conversationId is provided
    if not conversationId:
        return jsonify({"message": "Invalid conversation ID provided"}, status_code=400)

    # Initialize the SupabaseVectorStore instance
    vectordb = SupabaseVectorStore.from_documents({}, embeddings, client=supabase, user_id="")

    # Delete the chat based on conversationId
    try:
        vectordb.delete(ids=[conversationId])
        return jsonify({"message": "Chat deleted successfully"})
    except Exception as e:
        return jsonify({"message": f"Error deleting chat: {str(e)}"}, status_code=500)




@app.route('/drop', methods=['POST'])
def drop():
        # Initialize Dropbox API client
        with ApiClient(configuration) as api_client:
            signature_request_api = apis.SignatureRequestApi(api_client)
          # Parse JSON data from the request body
            request_data = request.get_json()
            text_data = request_data["chat"]
            pdf_file_path = text_to_pdf(text_data)

            # Extract signer email addresses from the request data
            signer_1_email = request_data["signer_1_email"]
            signer_2_email = request_data["signer_2_email"]
            title = request_data["title"]
            subject = request_data["subject"]
            message = request_data["message"]
            cc_email_addresses = request_data["cc_email_addresses"]  # Retrieve cc_email_addresses as specified in the request data

        
            # Define signers and other options
            signer_1 = models.SubSignatureRequestSigner(
                email_address=signer_1_email,
                name=signer_1_email,
                order=0,
            )

            signer_2 = models.SubSignatureRequestSigner(
                email_address=signer_2_email,
                name=signer_2_email,
                order=1,
            )

            signing_options = models.SubSigningOptions(
                draw=True,
                type=True,
                upload=True,
                phone=True,
                default_type="draw",
            )

            field_options = models.SubFieldOptions(
                date_format="DD - MM - YYYY",
            )

            data = models.SignatureRequestSendRequest(
                title=title,
                subject=subject,
                message=message,
                signers=[signer_1, signer_2],
                cc_email_addresses=cc_email_addresses,
                files=[open(pdf_file_path, "rb")],  # Use the generated PDF
                metadata={
                    "custom_id": 1234,
                    "custom_text": "NDA #9",
                },
                signing_options=signing_options,
                field_options=field_options,
                test_mode=True,
            )
            try:

            # Send a signature request
                response = signature_request_api.signature_request_send(data)
                # print(response)

                return 'Check your inbox on your email for signing', 200

            except ApiException as e:
                print("Exception when calling Dropbox Sign API: %s\n" % e)
                return jsonify({'error': str(e)}, status_code=500)



@app.route('/update' , methods = ["POST"] )
def saveId():
    data = request.get_json()


    google_id = data["google_id"]
    conv_id = data["conv_id"]
    response = data["response"]

    data_to_upsert = {
        "googleid": google_id,
        "conv_id": conv_id,
        "response": response
    }
    try:
        # Attempt to upsert the data into the "Con" table
        supabase.from_("demo").upsert([data_to_upsert]).execute()
        return jsonify({"message": "Data upserted successfully"})
    except Exception as e:
        # Handle the exception and provide an appropriate error response
        return jsonify({"error": str(e)}), 500  # HTTP 500 Inter




@app.route('/get_conversations/<google_id>', methods=["GET"])
def getConversations(google_id):
    try:
        # Fetch data from the "demo" table based on the provided Google ID
        query = supabase.from_("demo").select("conv_id, response").eq("googleid", google_id)
        response = query.execute()

        # Check if the response contains data
        if response.data:
            rows = response.data
            # print(rows)
            # Create a dictionary to store conv_id as keys and lists of responses as values
            conv_id_responses = {}
            for row in rows:
                conv_id = row["conv_id"]
                response = row["response"]
                if conv_id not in conv_id_responses:
                    conv_id_responses[conv_id] = []
                conv_id_responses[conv_id].append(response)

            return jsonify(conv_id_responses)
    
        else:
            return jsonify({})  # No data found, return an empty dictionary
    except Exception as e:
        return jsonify({"error": str(e)}), 500  # HTTP 500 Internal Server Error for failure

@app.route('/chat', methods=['POST'])
def api():
    data = request.get_json()
    # input_message is the actual data, the data mime type is specified in type
    input_message = data['prompt']

 
    # ai_id is the id of the ai example GPT4 or GPT3.5 or LLAMA etc 
    conv_id= data["conversationId"]

   
    
    response = handle_message(input_message, conv_id,internet=False,assesment=False,spell=False)


    return jsonify({'response': response})

@app.route('/spell', methods=['POST'])
def spell():
    data = request.get_json()
    # input_message is the actual data, the data mime type is specified in type
    input_message = data['prompt']

 
    # ai_id is the id of the ai example GPT4 or GPT3.5 or LLAMA etc 
    conv_id= data["conversationId"]

   
    
    response = handle_message(input_message, conv_id,internet=False,spell=True)


    return jsonify({'response': response})

@app.route('/assessment', methods=['POST'])
def assessment():
    data = request.get_json()
    # input_message is the actual data, the data mime type is specified in type
    input_message = data['prompt']

 
    # ai_id is the id of the ai example GPT4 or GPT3.5 or LLAMA etc 
    conv_id= data["conversationId"]

   
    
    response = handle_message(input_message, conv_id,internet=False,assesment=True,spell=False)


    return jsonify({'response': response})

@app.route('/chat-internet', methods=['POST'])
def internet():
    data = request.get_json()
    # input_message is the actual data, the data mime type is specified in type
    input_message = data['prompt']

 
    # ai_id is the id of the ai example GPT4 or GPT3.5 or LLAMA etc 
    conv_id= data["conversationId"]

   
    
    response = handle_message(input_message, conv_id,internet=True,assesment=False,spell=False)


    return jsonify({'response': response})





if __name__ == '__main__':
    app.run(host='0.0.0.0', port=os.environ.get('PORT', 4000), debug=False)
