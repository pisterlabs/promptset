import os
import time
import openai
import gradio as gr
import requests
import pandas as pd
from dotenv import load_dotenv
from weaviate.client import Client
from openai.embeddings_utils import get_embedding
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

weaviate_url=""
client=None
product_name=""
product_desc=""
openai

# Load Environment Variables
def load_env_variables():
    global weaviate_url
    global openai

    load_dotenv()
    openai.api_key=os.getenv("OPENAI_API_KEY")
    weaviate_url=os.getenv("WEAVIATE_URL")
    
    print('\n##########################################################')
    print('>>>>>>>>>>>>>  Displaying Environment Variables <<<<<<<<<<')
    print('##########################################################')
    print('Loaded OpenAI API Key     -> '+os.getenv("OPENAI_API_KEY"))
    print('Loaded Weaviate URL       -> '+os.getenv("WEAVIATE_URL"))
    print('##########################################################\n')

# Create Weaviate Connection
def weaviate_client():
    global client

    try:
        client = Client(url=weaviate_url, timeout_config=(3.05, 9.1))
        print("Weaviate client connected successfully!")
    except requests.exceptions.ConnectionError:
        print("Failed to connect to the Weaviate instance.")

# Create Product Table
def create_product_db():
    # Define the class "Product" with properties name,description
    product_class = {
                        "classes": [{
                            "class": "Product",
                            "description": "Store Product Names and Description",
                            "vectorizer": "none",
                            "properties": [
                                {
                                    "name": "name",
                                    "dataType": ["text"],
                                    "description": "Product Name"
                                },
                                {
                                    "name": "description",
                                    "dataType": ["text"],
                                    "description": "Product Description"
                                }    
                            ]
                        }]
                    }

    # Create the class in Weaviate
    try:
        response = client.schema.create(product_class)
        print("Class 'Product' created successfully!")
    except Exception as e:
        print(f"Failed to create class 'Product': {e}")

# Add Rows in Product Table
def add_product_data(product_name,product_desc):
    global weaviate_url
    global client

    # Check if Product Class is Available, else create
    try:
        client.schema.get("Product")
        print("Class 'Product' already exists!")
    except Exception as e:
        print(f"Error Verifying Class Product : {e}")
        create_product_db()
    
    # Check if data exists based on input - product_name
    data_object = {
        "name": product_name,
        "description": product_desc
    }
    where_filter = {
                        "path": ["name"],
                        "operator": "Equal",
                        "valueString": product_name
                   }

    query_result = (
                    client.query
                    .get("Product", "name")
                    .with_where(where_filter)
                    .do()
                )
    
    print("Product Table Query Result - "+str(query_result))
    if len(query_result['data']['Get']['Product']) == 0:
        client.data_object.create(data_object, class_name="Product")
    else:
        raise Exception("Product data already exits")

# Convert to CamelCase
def convert_to_camel_case(string):
    words = string.split('_')
    camel_case_words = [word.capitalize() for word in words]
    return ''.join(camel_case_words)

# Creating Training UM Tables
def create_kb_um_db():
    global product_name
    global product_desc

    product_class_name_camel_case = convert_to_camel_case(str(product_name+"_um"))
    print("Create UM Artefact of "+product_class_name_camel_case)

    # Define the class with `ProductUm` to store user manual details
    product_um =    {
                        "classes": [{
                            "class": product_class_name_camel_case,
                            "description": "Vector store of "+product_name+" user manual",
                            "vectorizer": "none",
                            "properties": [
                                {
                                    "name": "content",
                                    "dataType": ["text"],
                                    "description": "Store product "+product_name+" user manual details"
                                }  
                            ]
                        }]
                    }
    
    # Create the class in Weaviate
    try:
        response = client.schema.create(product_um)
        print("Class '"+str(product_um)+"' created successfully!")
    except Exception as e:
        print(f"Failed to create class '"+str(product_um)+"': {e}")
        

# Creating Training Mapping Tables
def create_kb_mapping_db():
    global product_name
    global product_desc

    product_class_name_camel_case = convert_to_camel_case(str(product_name+"_mapping"))
    print("Create Mapping Artefact of "+product_class_name_camel_case)

    # Define the class with `ProductMapping` to store user manual details
    product_mapping =    {
                        "classes": [{
                            "class": product_class_name_camel_case,
                            "description": "Vector store of "+product_name+" mapping",
                            "vectorizer": "none",
                            "properties": [
                                {
                                    "name": "key",
                                    "dataType": ["text"],
                                    "description": "Key Column"
                                },
                                {
                                    "name": "description",
                                    "dataType": ["text"],
                                    "description": "Description of Master Table Key Data"
                                }  
                            ]
                        }]
                    }
    
    # Create the class in Weaviate
    try:
        response = client.schema.create(product_mapping)
        print("Class '"+str(product_mapping)+"' created successfully!")
    except Exception as e:
        print(f"Failed to create class '"+str(product_mapping)+"': {e}")

# Create OpenAI Embedding
def create_openai_embeddings(text):
    print("Creating embedding for text"+ text)

    # Updating Embedding
    retry_attempts = 3
    retry_interval = 65

    # Create OpenAI embeddings
    for attempt in range(retry_attempts):
        try:
            embedding = get_embedding(text, engine="text-embedding-ada-002")
            return embedding
        except Exception as e:
            time.sleep(retry_interval)
            print(str(e))

# Store Object into `ProductUm`
def store_pdf_weaviate(new_embedding):
    global client
    global product_name

    print("Storing data into Weaviate")

    for item in new_embedding:
        data_object = {
            "content": item['text']
        }
        
        #Add the object to Weaviate
        client.data_object.create(data_object, class_name=convert_to_camel_case(str(product_name+"_um")),vector=item['embedding'])

# Extract and Store `ProductUm`
def extract_text_from_pdf(file):
    print("Extracting text from pdf")

    # Text Splitter
    text_splitter = CharacterTextSplitter(    
        chunk_size = 1000,
        chunk_overlap  = 0,
        length_function = len,
    )

    # Read the PDF file page by page
    reader = PdfReader(file.name)

    # Process each page individually
    for i, page in enumerate(reader.pages):
        new_embedding = []

        text = page.extract_text()
        print('--------Page Content-------')
        print(text)
        if text:
            
            # Split the text into smaller chunks
            chunks = text_splitter.split_text(text)

            # Process each chunk individually
            for chunk in chunks:
                new_embedding.append({
                    'text': chunk,
                    'embedding': create_openai_embeddings(chunk)
                })

        # Storing Data
        store_pdf_weaviate(new_embedding)

# Wrapper to Extract & Store PDF Data
def process_pdf_file(file):
    print("Processing PDF file")
    
    # Read the data from the pdf and store to vector
    extract_text_from_pdf(file)

# Train Data UM via PDF File Uploaded
def add_kb_um_data():
    global product_name
    global product_desc

    product_class_name_camel_case = convert_to_camel_case(str(product_name+"_um"))

    # Check if `ProductUm` Class is Available, else create
    try:
        client.schema.get(product_class_name_camel_case)
        raise Exception("Class '"+str(product_class_name_camel_case)+"' already exists!")
    except Exception as e:
        if "already exists!" in str(e):
            raise Exception("Class '"+str(product_class_name_camel_case)+"' already exists!")
        elif "404" in str(e):
            create_kb_um_db()
        else:
             raise Exception(f"Error Verifying Class {str(product_class_name_camel_case)} : {e}")

# Store Object into `ProductUm`
def store_xlsx_weaviate(new_embedding):
    global client
    global product_name
    
    print("Storing data into Weaviate")

    for item in new_embedding:
        data_object = {
            "key": item['key'],
            "description": item['description']
        }

        #Add the object to Weaviate
        client.data_object.create(data_object, class_name=convert_to_camel_case(str(product_name+"_mapping")),vector=item['embedding'])


def extract_text_from_xlsx(file):
    # Replace 'file_path.xlsx' with the actual path of your Excel file
    file_path = file.name

    # Read all tabs from the Excel file into a dictionary of dataframes
    dfs = pd.read_excel(file_path, sheet_name=None)

    # Create an empty dictionary to store the combined values
    combined_values = {}

    # Loop through each dataframe in the dictionary
    for sheet_name, df in dfs.items():
        # Get the column names and hints from the dataframe
        column_names = df['Column Name'].tolist()
        hints = df['Hint'].tolist()

        # Combine the values and add them to the dictionary
        combined_values.update({f"{sheet_name}.{column_names}": f"{column_names} {hint}" for column_names, hint in zip(column_names, hints)})

    # Print the combined values
    for key, value in combined_values.items():
        new_embedding = []

        print(f"Key: {key}")
        print(f"Value: {value}")
        print("-------------------------")
        new_embedding.append({
            'key':key,
            'description': value,
            'embedding': create_openai_embeddings(value)
        })

        store_xlsx_weaviate(new_embedding)


# Wrapper to Extract & Store XLSX Data
def process_xlsx_file(file):
    print("Processing Mapping xlsx file")
    extract_text_from_xlsx(file)


# Train Data Mapping via excel File Uploaded
def add_kb_mapping_data():
    global product_name
    global product_desc

    product_class_name_camel_case = convert_to_camel_case(str(product_name+"_mapping"))

    # Check if `ProductMapping` Class is Available, else create
    try:
        client.schema.get(product_class_name_camel_case)
        raise Exception("Class '"+str(product_class_name_camel_case)+"' already exists!")
    except Exception as e:
        if "already exists!" in str(e):
            raise Exception("Class '"+str(product_class_name_camel_case)+"' already exists!")
        elif "404" in str(e):
            create_kb_mapping_db()
        else:
             raise Exception(f"Error Verifying Class {str(product_class_name_camel_case)} : {e}")

# Text Prompt
def add_text(history, text):
    global product_name
    global product_desc

    if "Create New Knowledge Base" in text:
        history = history + [(text, "Enter Product Name")]
    elif "Product Name" in text:
        dash_index = text.index('-')
        comma_index = text.index(',')

        # Sample Prompt text= Product Name - OFSLL, Oracle Financial Lending and Leasing
        product_name = text[dash_index + 1:comma_index].strip() 
        product_desc = text[comma_index + 1:].strip()
        print("product_name - "+product_name+",product_desc - "+product_desc)
        try:
            add_product_data(product_name,product_desc)
            history = history + [(text,"Created New Knowledge Base for - "+product_name)]
        except Exception as e:
            if "Product data already exits" in str(e):
                print(f"Failed to add object to class 'Product': {e}")
                history = history + [(text,"<strong style='color:red'>Knowledge Base for - "+product_name+", Already Exists</strong>")]
            else:
                history = history + [(text,f"<strong style='color:red'>Error Processing Request - {e}</strong>")]
    else:
        history = history + [(text, "Hello There !!!<br><br> Please Enter:  <br>1) Create New Knowledge Base <br>2) Update Knowledge Base")]
    return history, ""

# PDF Confirmation Message
def confirmation_pdf_file(history):
    history = history + [(None, "Processing User Manual PDF Completed")]
    return history

# Excel Confirmation Message
def confirmation_excel_file(history):
    history = history + [(None, "Processing Mapping Excel Data Completed")]
    return history

# PDF Upload Of User Manual
def add_pdf_file(history, file):
    print("User uploaded PDF File")
    history = history + [(None, "User Manual PDF Uploaded Successfully")]
    
    try:
        add_kb_um_data()
        print("Uploaded file"+file.name)
        process_pdf_file(file)
    except Exception as e:
        if "already exists!" in str(e):
            history = history + [(None,"<strong style='color:red'>UM Knowledge Base for - "+product_name+", Already Exists</strong>")]
        else:
            history = history + [(None,f"<strong style='color:red'>Error Processing Request - {e}</strong>")]

    return history

# Excel Upload Of Migration Data Model
def add_xlsx_file(history, file):
    print("User uploaded Excel File")
    history = history + [(None, "Mapping Excel Data Uploaded Successfully")]

    try:
        add_kb_mapping_data()
        print("Uploaded file"+file.name)
        process_xlsx_file(file)
    except Exception as e:
        if "already exists!" in str(e):
            history = history + [(None,"<strong style='color:red'>Mapping Knowledge Base for - "+product_name+", Already Exists</strong>")]
        else:
            history = history + [(None,f"<strong style='color:red'>Error Processing Request - {e}</strong>")]

    return history

# Start of Program - Main
def main():
    print("\nStarted Knowledge Base Application")

    load_env_variables()
    weaviate_client()

    # Designing Chatbot UI
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot([], elem_id="chatbot").style(height=750)

        with gr.Row():
            with gr.Column(scale=0.85):
                txt = gr.Textbox(
                    show_label=False,
                    placeholder="Send me a message",
                ).style(container=False)
            with gr.Column(scale=0.075, min_width=0):
                btn_pdf = gr.UploadButton("📄", file_types=[".pdf"])
            with gr.Column(scale=0.075, min_width=0):
                btn_excel = gr.UploadButton("📊", file_types=[".xlsx"])

        txt.submit(add_text, [chatbot, txt], [chatbot,txt])
        btn_pdf.upload(add_pdf_file, [chatbot, btn_pdf], [chatbot]).then(
                        confirmation_pdf_file, chatbot, chatbot
                      )
        btn_excel.upload(add_xlsx_file, [chatbot, btn_excel], [chatbot]).then(
                        confirmation_excel_file, chatbot, chatbot
                      )

    demo.launch(server_name="0.0.0.0")

if __name__ == '__main__':
    main()