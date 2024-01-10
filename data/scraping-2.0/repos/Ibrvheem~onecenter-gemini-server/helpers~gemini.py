import os

import google.generativeai as genai

from app.partner.model import Partner
from app.user.model import User

GOOGLE_API_KEY=os.getenv('GOOGLE_API_KEY')

genai.configure(api_key=GOOGLE_API_KEY)


# from langchain_core.messages import HumanMessage, AIMessage
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import pinecone
from langchain.vectorstores.pinecone import Pinecone
# from langchain.schema.output_parser import StrOutputParser
# from langchain.schema.runnable import RunnableMap
# from langchain.prompts import ChatPromptTemplate

from langchain.document_loaders import OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# llm = ChatGoogleGenerativeAI(model="gemini-pro",
#                              temperature=0.7)

# def dotest():
#     result = llm.invoke("What is a LLM?")
#     print(result.content)

def qa_chain(question = "What happened to the oceangate submersible?", history=[], partner: Partner = Partner(), user: User = User()):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    pinecone.init(
        api_key=os.getenv('PINECONE_API_KEY'),
        environment=os.getenv('PINECONE_API_ENV'),
    )

    docsearch = Pinecone.from_existing_index(index_name=partner.identity, embedding=embeddings)
    retriever=docsearch.as_retriever()

    system_message = f"""
            "You are a helpful customer support agent."
            "You provide assistant to callers about {partner.name}"
            "You can ask questions to help you understand and diagnose the problem."
            "If you are unsure of how to help, you can suggest the client to go to the nearest {partner.name} office."
            "Try to sound as human as possible"
            "Make your responses as concise as possible"
            "Your response must be in plain text"
            "If a user is asking for their order status, check the database for any order that belongs to their phone number"
            "My name is {user.name}"
            "My phone number is {user.phone}"
            """

    model = genai.GenerativeModel('gemini-pro')

    messages = [
        {
            'role':'user',
            'parts': [system_message]
            }
    ]
    for h in history:
        messages[0]['parts'].append(h.question)
        messages[0]['parts'].append(h.answer)
    
    retrieved = retriever.get_relevant_documents(question)
    context = "\n".join([document.page_content for document in retrieved])
    database = """
[
  {
    "customerPhoneNumber": "08012345678",
    "productName": "Litetouch Smart Lamp",
    "orderId": "LT-OD-456789",
    "deliveryStatus": "En route",
    "deliveryAgent": "John Doe",
    "estimatedDeliveryTime": "2023-12-27T17:30:00 WAT",
    "customerAddress": {
      "street": "123 Main Street",
      "city": "Abuja",
      "state": "Federal Capital Territory",
      "country": "Nigeria"
    }
  },
  {
    "customerPhoneNumber": "08098765432",
    "productName": "Litetouch Wireless Headphones",
    "orderId": "LT-OD-123456",
    "deliveryStatus": "Pending",
    "deliveryAgent": null,
    "estimatedDeliveryTime": null,
    "customerAddress": {
      "street": "456 Oak Avenue",
      "city": "Lagos",
      "state": "Lagos State",
      "country": "Nigeria"
    }
  },
  {
    "customerPhoneNumber": "08035401479",
    "productName": "Litetouch Fitness Tracker",
    "orderId": "LT-OD-789012",
    "deliveryStatus": "Delivered",
    "deliveryAgent": "Jane Smith",
    "estimatedDeliveryTime": "2023-12-27T16:45:00 WAT",
    "customerAddress": {
      "street": "789 Elm Street",
      "city": "Ibadan",
      "state": "Oyo State",
      "country": "Nigeria"
    }
  },
  {
    "customerPhoneNumber": "08162577778",
    "productName": "Litetouch Smart Speaker",
    "orderId": "LT-OD-345678",
    "deliveryStatus": "En route",
    "deliveryAgent": "Michael Johnson",
    "estimatedDeliveryTime": "2023-12-27T18:00:00 WAT",
    "customerAddress": {
      "street": "101 Pine Street",
      "city": "Kano",
      "state": "Kano State",
      "country": "Nigeria"
    }
  },
  {
    "customerPhoneNumber": "08163089308",
    "productName": "Litetouch Portable Charger",
    "orderId": "LT-OD-901234",
    "deliveryStatus": "Out for delivery",
    "deliveryAgent": "Peter Jackson",
    "estimatedDeliveryTime": "2023-12-27T17:15:00 WAT",
    "customerAddress": {
      "street": "222 Birch Street",
      "city": "Port Harcourt",
      "state": "Rivers State",
      "country": "Nigeria"
    }
  },
  {
    "customerPhoneNumber": "07039012508",
    "productName": "Litetouch Wireless Keyboard",
    "orderId": "LT-OD-567890",
    "deliveryStatus": "Pending",
    "deliveryAgent": null,
    "estimatedDeliveryTime": null,
    "customerAddress": {
      "street": "333 Maple Street",
      "city": "Enugu",
      "state": "Enugu State",
      "country": "Nigeria"
    }
  },
  {
    "customerPhoneNumber": "08039012509",
    "productName": "Litetouch Ergonomic Mouse",
    "orderId": "LT-OD-012345",
    "deliveryStatus": "En route",
    "deliveryAgent": "Mary Brown",
    "estimatedDeliveryTime": "2023-12-27T16:30:00 WAT",
    "customerAddress": {
      "street": "444 Cedar Street",
      "

"""

    messages[0]['parts'].append(f"Based on our conversation and the context below: {question}\n Context: {context}\n\n Database:{database}")
    
    response = model.generate_content(messages)
    return response.text

def pinecone_train_with_resource(resource_url, partner_identity):
    loader = OnlinePDFLoader(resource_url)

    data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)
    

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    pinecone.init(
        api_key=os.getenv('PINECONE_API_KEY'),
        environment=os.getenv('PINECONE_API_ENV'),
    )
    
    if partner_identity not in pinecone.list_indexes():
        # we create a new index
        pinecone.create_index(
        name=partner_identity,
        metric='cosine',
        dimension=1536
        )

    Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=partner_identity)