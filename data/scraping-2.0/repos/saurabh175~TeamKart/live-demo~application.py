from flask import Flask, request, render_template, jsonify
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import os
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.document_loaders.json_loader import JSONLoader
import json
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from api import findKNearestItems
from item import ProductEncoder, create_product_object

app = Flask(__name__)

rapidapi_key = "d83f4bfe89msh2c35c0f026b5666p1bfa94jsnb9eba71c9060"  # Replace with your RapidAPI key
rapidapi_endpoint = "https://gpt-chat.p.rapidapi.com/v1/gpt-3.5-turbo"


# openai_api_key = "sk-d83f4bfe89msh2c35c0f026b5666p1bfa94jsnb9eba71c9060"

# llm = OpenAI(openai_api_key=openai_api_key, temperature=0)
# with open("faq.txt", "r", encoding="utf-8") as file:
#     content = file.read()
# with open("faq.txt", "r", encoding="utf-8", errors="ignore") as file:
#     content = file.read()


instructions = '''

You are a online shopping assistant with two tasks - answering questions about inventory and company policy or generating product bundles. 
Specifically, you are an online shopping assistant meant to guide users through the website of the company 'Son of a Tailor'.

It is your job to deduce whether the user is asking a question about the company's policy or is asking about potential products to buy.
For each case, follow the follow instructions respectively.

START OF INSTRUCTIONS

If user is asking a general FAQ question about policies, DO NOT RECCOMEND ANY PRODUCTS out of the inventory. After directly responding to the 
question asked by the user about policies, terminate your answer. 

1. If the user is asking questions about the company's policies or general information
Answer the question using the company's FAQs data. 
Your response should follow the following format (under no circumstances should you recommend a product in this case):
FAQ Response: <response>

If the question is about products, follow the below protocol:
    
2. The user is looking for products to buy.
If you do not know the exact name of a product or it does not exist within the company's inventory, tell them that we do not offer it at the moment. Do not make up or reference products that are not directly from the data provided. 
Only provide the DKK and USD prices unless specified for a different currency.
Don't just build outfits based on your general knowledge - only base your oufits on the product list you were given. Don't make up the names of products that you don't have access to. We only sell tops. 
If a product has more than one color available, suggest a color but also say we have more colors.
Verify that the product bundles you are generating are adequate and are of the same type as the request being made and fit the appropriate criteria
If the user says some gibberish or something that is not a question or doesn't make sense, say that they have to clarify and you don't understand by saying I'm sorry, I don't understand. Could you please clarify what you are asking?
Keep your responses under 200 word. At the end of your response, list out each product that you chose from, why you chose that product, and confirm that the product was found in the list of products we inputted.
If the user provides you details about why they need something (region, reason, age), cater your results to this preference.
Your response should be in the following format:
    - <product >
    - <product >
    - <product >
    ...
<reasoning >

Only provde the title of the product and price, no other information. Do not provide materials unless asked for. 
Keep in mind the context of what the user has said in the conversation below when giving answers to additional questions.
If the user is not asking or requesting products to buy, just answer their question without recommending any clothing items. Parse through the FAQs to get relevant information.

END OF INSTRUCTIONS

Current Conversation:
{history}

Here are the products related to the current query: {input}

The <reasoning> should come after the product listing and should be brief. Keep the word count low and concise. 

AI Assistant:'''


PROMPT = PromptTemplate(
    input_variables=["history", "input"], template=instructions
)


conversation = ConversationChain(
    prompt=PROMPT,
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory(ai_prefix="AI Assistant")
)

# chrome://net-internals/#sockets


@app.route('/', methods=['GET', 'POST'])
def index():
    return render_template('popup.html')


# @app.route('/get_parameter', methods=['GET'])
# def get_parameter():
#     input_data = request.args.get('input')
#     nearest_items = findKNearestItems(input_data, 5)
#     products = process_products(nearest_items)
#     input_products = product_template(products)
#     llm_response = conversation.predict(
#         input=input_products + "\n Customer: " + input_data)
#     print(llm_response)
#     res = {"data": products, "result": llm_response}
#     return jsonify(res)

@app.route('/get_parameter', methods=['GET'])
def get_parameter():
    input_data = request.args.get('input')

    # Send request to RapidAPI
    headers = {
        "X-RapidAPI-Host": "gpt-chat.p.rapidapi.com",
        "X-RapidAPI-Key": rapidapi_key,
        "Content-Type": "application/json",
    }

    payload = {
        "messages": [{"role": "user", "content": input_data}],
        "model": "gpt-3.5-turbo",
        "max_tokens": 100,
        "temperature": 0.9,
    }

    response = request.post(rapidapi_endpoint, headers=headers, json=payload)
    response_data = response.json()
    llm_response = response_data["choices"][0]["message"]["content"]

    # Process llm_response and generate products
    nearest_items = findKNearestItems(input_data, 5)
    products = process_products(nearest_items)
    input_products = product_template(products)

    # Generate AI response using conversation.predict
    conversation_history = ""  # Add your conversation history here
    llm_response = conversation.predict(
        input=input_products + "\n Customer: " + input_data + "\n" + conversation_history
    )
    print(llm_response)

    res = {"data": products, "result": llm_response}
    return jsonify(res)


# turns each string into a product object and converts to json


def process_products(inputs):
    products = []
    for item in inputs:
        product = create_product_object(item[0])
        product_json = json.dumps(product, cls=ProductEncoder)
        product_dict = json.loads(product_json)
        products.append(product_dict)

    return products


def product_template(products):
    res = []
    for p in products:
        res.append(
            {
                'name': p['title'],
                'desc': p['desc']
            }
        )
    return str(res)


if __name__ == "__main__":
    # Change the host and port as desired
    app.run(host='localhost', port=9000)
