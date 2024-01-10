import os
import logging
import click
import torch, time
import utils
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler  # for streaming response
from langchain.callbacks.manager import CallbackManager
from subprocess import call
import shutil
import multiprocessing
import signal

from tensorflow.keras.utils import load_img, img_to_array 

from PIL import Image


import os

# os.environ["HTTPS_PROXY"] = "http://localhost:8080"

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

from prompt_template_utils import get_prompt_template

# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from transformers import (
    GenerationConfig,
    pipeline,
)

from load_models import (
    load_quantized_model_gguf_ggml,
    load_quantized_model_qptq,
    load_full_model,
)

from constants import (
    EMBEDDING_MODEL_NAME,
    PERSIST_DIRECTORY,
    MODEL_ID,
    MODEL_BASENAME,
    MAX_NEW_TOKENS,
    MODELS_PATH,
    CHROMA_SETTINGS
)


import bs4
from bs4 import BeautifulSoup
import re,time,ast, requests
import pandas as pd
import numpy as np 
import os
import requests
import spacy

import socket
hostname = socket.gethostname()
print(hostname)

from googlesearch import search   

import cv2, sys
import pytesseract
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import matplotlib.pyplot as plt
from spellchecker import SpellChecker
import csv

import requests
from bs4 import BeautifulSoup
from fpdf import FPDF
from io import BytesIO
from PIL import Image    #install pyspellcher, fpdf, langchain
import os

from keras.preprocessing import image
from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import numpy as np


from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors


nlp = None
        
try:
    nlp = spacy.load("en_core_web_sm")
    print('inside try...*****\n')
except:
    nlp = spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    print('inside except...******\n')

print('Is NLP none? ',nlp == None,'\n')
    
def process_query(query):
    
    start_X = time.time()

    print('Got query : ',query,'\n\n')
    
    # to search 
    # query = "Public safety picture of Manhattan Beach, LA, how safe for students?"

    links = []
    
    # print('@@@@@ Length of the search results : ',len(list(search(query))),'\n\n')
    
    for j in search(query,tld="co.in", num=10, stop=10, pause=2): 
        # print('@@@@@@@ Got one...\n', j)
        # time.sleep(5)
        links.append(j) 
    
    
    # for j in search(query): 
    #     # print('@@@@@@@ Got one...\n', j)
    #     time.sleep(5)
    #     links.append(j) 

    ## scraping each website, getting the texts and the images in them..
    
    ### trying out my code to filter results taking too long to load
    
    all_website_content = []
    
    for url in links:

        try:
            response = requests.get(url,timeout=3)
            # print('After response...')
            soup = BeautifulSoup(response.text, 'html.parser')
            # print('After soup...')
            # Extract text content from the parsed HTML
            
            text_content = soup.get_text().strip()
            all_website_content.append(text_content)

            # print("Text content from", url, ":\n", text_content.strip()[:100])
            print("-" * 50)

            # else:
            #     print(f"Failed to retrieve content from {url}. Status code: {response.status_code}")

        except Exception as e:
            print(f"An error occurred while processing {url}: {str(e)}")


    #saving the scraped data to the file

    # Name of the output PDF file
    pdf_file = "SOURCE_DOCUMENTS/scraped_data.pdf"

    doc = SimpleDocTemplate(pdf_file, pagesize=letter)

    story = []

    # Defining a style
    styles = getSampleStyleSheet()
    style = styles["Normal"]

    spacy_link_content = all_website_content[:]

    spacy_link_content = [i.replace('\n','').strip() for i in spacy_link_content]

    spacy_link_content = [i.replace('\t','') for i in spacy_link_content]

    for string in spacy_link_content:
        try:
            p = Paragraph(string, style)
            story.append(p)
            story.append(Paragraph("<br/>", style))
        except:
            print('Inside except')
            # print(f"String content: {string[:100]}")

    doc.build(story)

    print(f"PDF saved as {pdf_file}")

    ### extracting the image URLs and storing them in a lsit


    def extract_image_urls_from_webpage(url):
        
        image_urls = []
        
        try:
            response = requests.get(url)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                img_tags = soup.find_all('img')  # Find all image tags in the HTML

                for img in img_tags:
                    img_url = img.get('src')  # Get the 'src' attribute of the image tag
                    if img_url and (img_url.startswith('http') or img_url.startswith('https')):
                        image_urls.append(img_url)
        except:
            pass

        return image_urls


    all_image_urls = []
    
    for url in links[:5]:
        image_urls = extract_image_urls_from_webpage(url)
        all_image_urls.extend(image_urls)

    print('@@@@ Length of all_image_urls is : ',len(all_image_urls),'\n\n')
    
    ##The code below will scrape all images from the web pages and store them inside fodler named X. Please don't touch this

    def download_images(image_urls, folder_name):
        # Create the folder if it doesn't exist
        print('\n\n',len(image_urls),'@@@@@\n')
        print('Folder name where images will be stored : ',folder_name,'\n\n')
        
        shutil.rmtree(folder_name)
        
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            
        for index, url in enumerate(image_urls, start=1):
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    image_name = f'image_{index}.png'  # Modify the file extension if needed
                    image_path = os.path.join(folder_name, image_name)

                    with open(image_path, 'wb') as image_file:
                        image_file.write(response.content)

                    print(f"Downloaded: {image_name}")
                else:
                    print(f"Failed to download: {url}")

            except Exception as e:
                print(f"Error downloading {url}: {e}")


    start = time.time()
    folder_name = 'images'
    download_images(all_image_urls[:40], folder_name)
    end = time.time()

    print('Time taken to download images : ',end - start,'\n\n')

    ### now writing a code which will iterate through the imags in the folder and say which image is most closely related to the user query (for images without text)

    ###code to find matching images

    model = VGG16(weights='imagenet')

    # provide the folder path accordingly..

    folder_path = 'images/'

    image_objects = {}  # Dictionary to store image objects

    # Iterate through all files in the folder

    #for the images without texts

    start = time.time()

    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Filter by file extensions
            img_path = os.path.join(folder_path, filename)
            # print('Image path:', img_path)

            try:
                img = Image.open(img_path)
                img.verify()  # Check if the file is a valid image
                img = tf.keras.utils.load_img(img_path, target_size=(224, 224))  # VGG16 input size
                x = img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                preds = model.predict(x)

                decoded_preds = decode_predictions(preds, top=5)[0]  # Get top 3 predictions
                # print("Predictions:","for image : ",img_path)
                for i, (imagenet_id, label, score) in enumerate(decoded_preds):
                    print(f"{i + 1}: {label} ({score:.2f})")

                #code to store images as keys and values as the objects in it

                objects_detected = [label for (_, label, _) in decoded_preds]  # Extract labels

                # Store detected objects in the dictionary with filename as key
                image_objects[filename] = objects_detected

            except (IOError, SyntaxError) as e:
                # Skip over files that are not valid images
                # print(f"Skipped: {img_path} - Error: {e}")
                pass

    end = time.time()

    print('Time taken to process images without texts : ',end - start,'\n\n')

    ### the functiosn below all together show images most related to the user search query, remember this would work for images with texts


    #there can be images with texts, this is the code below for that..


    # Loading pre-trained MobileNetV2 model
    model = MobileNetV2(weights='imagenet', include_top=True)

    # Function to extract text using Tesseract OCR
    def extract_text_from_image(img):
        return pytesseract.image_to_string(img)

    def img_process(img_path):

        # Load and ?preprocess image
        img = image.load_img(img_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Predict objects in the image
        predictions = model.predict(x)
        decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=10)[0]

        # Store recognized objects and extracted text as sets
        recognized_objects = set()
        extracted_text = set()

        for _, label, _ in decoded_predictions:
            recognized_objects.add(label)

        # Convert image to OpenCV format for text extraction
        img_cv = cv2.imread(img_path)

        # Extract text from the image using Tesseract OCR
        text = extract_text_from_image(img_cv)
        extracted_text.add(text)

        # Print the sets of recognized objects and extracted text
        # print("Recognized Objects:")
        # print(recognized_objects)
        # print("\nExtracted Text:")
        # print(extracted_text)
        return extracted_text


    def discard_wrong_spellings(text):
        spell = SpellChecker()

        # Tokenize the text into words
        words = text.strip().split()

        # Get the set of misspelled words
        misspelled = spell.unknown(words)

        # print('Misspelled words are : ',misspelled)

        # Filter out words that are not misspelled
        correct_words = [word for word in words if word not in misspelled]

        return ' '.join(correct_words)

    def get_image_filenames(directory):
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']  # Add more extensions if needed
        image_files = []

        for root, dirs, files in os.walk(directory):
            for file in files:
                if any(file.lower().endswith(ext) for ext in image_extensions):
                    image_files.append(os.path.join(root, file))

        return image_files

    def get_text_for_image(folder, user_query):

        dict_img_words = dict()

            # Replace 'images' with the path to the directory
        image_directory = folder
        # image_directory = '/Users/soumyarn/USC/Fall_2023/DSCI_560/Project/downloaded_images/'

        image_filenames = get_image_filenames(image_directory)


        start = time.time()
        # Print the collected image filenames

        for filename in image_filenames:
            try:
                if 'checkpoints' not in filename:
                    extracted_text = img_process(filename)
                    valid_text = remove_special_alphanumeric_words(extracted_text)
                    string = " ".join(i for i in valid_text)
                    fnal_ans = discard_wrong_spellings(string)
                    dict_img_words[filename] = fnal_ans
            except:
                pass
        end = time.time()

        print('Time taken to process all images : ',end - start,'\n\n')

        for k,v in dict_img_words.items():
            if v:
                v = v.lower()
                dict_img_words[k] = v

        print('User query : ',user_query)

        values = list(dict_img_words.values())

        values.append(user_query)

        # Load a pre-trained Sentence Transformer model (you can choose other models too)
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

        value_embeddings = model.encode(values, convert_to_tensor=True)

        query_embedding = model.encode([user_query], convert_to_tensor=True)

        cosine_scores = util.pytorch_cos_sim(query_embedding, value_embeddings)

        cosine_scores = cosine_scores.cpu().numpy()

        similar_indices = cosine_scores.argsort(axis=1)[0][::-1]

        # Setting a similarity threshold (adjust as needed)
        threshold = 0.05

        similar_keys = []

        for i in list(similar_indices):
            if cosine_scores[0][i] > 0.03:
                try:
                    if 'checkpoints' not in list(dict_img_words.keys())[i]:
                        similar_keys.append((list(dict_img_words.keys())[i],cosine_scores[0][i]))
                except:
                    continue

        displayed_images = []
        x = set()

        for img_filename, sim_score in similar_keys[:10]:
            try:
                img = Image.open(img_filename)

                image_content = tuple(img.getdata())

                if image_content not in x:
                    # plt.imshow(img)
                    # plt.title(f"Similarity Score: {sim_score}")
                    # plt.axis('off') 
                    # plt.show()
                    x.add(image_content)
                    displayed_images.append((img_filename,sim_score))
            except FileNotFoundError:
                print(f"Image file {key} not found.")

        return displayed_images

    folder_path = 'images/'
    x = get_text_for_image(folder_path, query)   #provide the folder path accordingly

    #saving the images with detected injects to SOURCE_DOCUMENTS

    csv_filename = 'SOURCE_DOCUMENTS/image_objects.csv'

    print('Image objects dict : ',image_objects)


    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Image Filename', 'Detected Objects'])  # Write header

        for img_file, objects in image_objects.items():
            csv_writer.writerow([img_file, ', '.join(objects)])

    print(f"Data has been written to {csv_filename}")

    
    end_X = time.time()

#     print('Finished populating SOURCE_DOCUMENTS and fiding the most related ones....\n')

#     print('Time taken to find similar images : ',end_query - start_query,'\n\n')
    
    print('Total TIme taken : ',end_X - start_X,'\n\n')


def load_model(device_type, model_id, model_basename=None, LOGGING=logging):
    """
    Select a model for text generation using the HuggingFace library.
    If you are running this for the first time, it will download a model for you.
    subsequent runs will use the model from the disk.

    Args:
        device_type (str): Type of device to use, e.g., "cuda" for GPU or "cpu" for CPU.
        model_id (str): Identifier of the model to load from HuggingFace's model hub.
        model_basename (str, optional): Basename of the model if using quantized models.
            Defaults to None.

    Returns:
        HuggingFacePipeline: A pipeline object for text generation using the loaded model.

    Raises:
        ValueError: If an unsupported model or device type is provided.
    """
    logging.info(f"Loading Model: {model_id}, on: {device_type}")
    logging.info("This action can take a few minutes!")

    if model_basename is not None:
        if ".gguf" in model_basename.lower():
            llm = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
            return llm
        elif ".ggml" in model_basename.lower():
            model, tokenizer = load_quantized_model_gguf_ggml(model_id, model_basename, device_type, LOGGING)
        else:
            model, tokenizer = load_quantized_model_qptq(model_id, model_basename, device_type, LOGGING)
    else:
        model, tokenizer = load_full_model(model_id, model_basename, device_type, LOGGING)

    # Load configuration from the model to avoid warnings
    generation_config = GenerationConfig.from_pretrained(model_id)
    # see here for details:
    # https://huggingface.co/docs/transformers/
    # main_classes/text_generation#transformers.GenerationConfig.from_pretrained.returns

    # Create a pipeline for text generation
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=MAX_NEW_TOKENS,
        temperature=0.2,
        # top_p=0.95,
        repetition_penalty=1.15,
        generation_config=generation_config,
    )

    local_llm = HuggingFacePipeline(pipeline=pipe)
    logging.info("Local LLM Loaded")

    return local_llm


def retrieval_qa_pipline(device_type, use_history, promptTemplate_type="llama"):
    """
    Initializes and returns a retrieval-based Question Answering (QA) pipeline.

    This function sets up a QA system that retrieves relevant information using embeddings
    from the HuggingFace library. It then answers questions based on the retrieved information.

    Parameters:
    - device_type (str): Specifies the type of device where the model will run, e.g., 'cpu', 'cuda', etc.
    - use_history (bool): Flag to determine whether to use chat history or not.

    Returns:
    - RetrievalQA: An initialized retrieval-based QA system.

    Notes:
    - The function uses embeddings from the HuggingFace library, either instruction-based or regular.
    - The Chroma class is used to load a vector store containing pre-computed embeddings.
    - The retriever fetches relevant documents or data based on a query.
    - The prompt and memory, obtained from the `get_prompt_template` function, might be used in the QA system.
    - The model is loaded onto the specified device using its ID and basename.
    - The QA system retrieves relevant documents using the retriever and then answers questions based on those documents.
    """

    embeddings = HuggingFaceInstructEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": device_type})
    # uncomment the following line if you used HuggingFaceEmbeddings in the ingest.py
    # embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    # load the vectorstore
    db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings,
        client_settings=CHROMA_SETTINGS
    )
    retriever = db.as_retriever()

    # get the prompt template and memory if set by the user.
    prompt, memory = get_prompt_template(promptTemplate_type=promptTemplate_type, history=use_history)

    # load the llm pipeline
    llm = load_model(device_type, model_id=MODEL_ID, model_basename=MODEL_BASENAME, LOGGING=logging)
    
    print('******* Is LLM None? ',llm == None,'\n\n')

    if use_history:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
            retriever=retriever,
            return_source_documents=True,  # verbose=True,
            callbacks=callback_manager,
            chain_type_kwargs={"prompt": prompt, "memory": memory},
        )
    else:
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # try other chains types as well. refine, map_reduce, map_rerank
            retriever=retriever,
            return_source_documents=True,  # verbose=True,
            callbacks=callback_manager,
            chain_type_kwargs={
                "prompt": prompt,
            },
        )

    return qa


# chose device typ to run on as well as to show source documents.
@click.command()
@click.option(
    "--device_type",
    default="cuda" if torch.cuda.is_available() else "cpu",
    type=click.Choice(
        [
            "cpu",
            "cuda",
            "ipu",
            "xpu",
            "mkldnn",
            "opengl",
            "opencl",
            "ideep",
            "hip",
            "ve",
            "fpga",
            "ort",
            "xla",
            "lazy",
            "vulkan",
            "mps",
            "meta",
            "hpu",
            "mtia",
        ],
    ),
    help="Device to run on. (Default is cuda)",
)
@click.option(
    "--show_sources",
    "-s",
    is_flag=True,
    help="Show sources along with answers (Default is False)",
)
@click.option(
    "--use_history",
    "-h",
    is_flag=True,
    help="Use history (Default is False)",
)
@click.option(
    "--model_type",
    default="llama",
    type=click.Choice(
        ["llama", "mistral", "non_llama"],
    ),
    help="model type, llama, mistral or non_llama",
)
@click.option(
    "--save_qa",
    is_flag=True,
    help="whether to save Q&A pairs to a CSV file (Default is False)",
)

def main(device_type, show_sources, use_history, model_type, save_qa):
    """
    Implements the main information retrieval task for a localGPT.

    This function sets up the QA system by loading the necessary embeddings, vectorstore, and LLM model.
    It then enters an interactive loop where the user can input queries and receive answers. Optionally,
    the source documents used to derive the answers can also be displayed.

    Parameters:
    - device_type (str): Specifies the type of device where the model will run, e.g., 'cpu', 'mps', 'cuda', etc.
    - show_sources (bool): Flag to determine whether to display the source documents used for answering.
    - use_history (bool): Flag to determine whether to use chat history or not.

    Notes:
    - Logging information includes the device type, whether source documents are displayed, and the use of history.
    - If the models directory does not exist, it creates a new one to store models.
    - The user can exit the interactive loop by entering "exit".
    - The source documents are displayed if the show_sources flag is set to True.

    """
    
    # print(device_type, show_sources, use_history, model_type, save_qa)
    logging.info(f"Running on: {device_type}")
    logging.info(f"Display Source Documents set to: {show_sources}")
    logging.info(f"Use history set to: {use_history}")

    # check if models directory do not exist, create a new one and store models here.
    if not os.path.exists(MODELS_PATH):
        os.mkdir(MODELS_PATH)

    print('@@@@@ All drvice values : ',device_type, show_sources, use_history, model_type, save_qa,'\n\n')     
       
    qa = retrieval_qa_pipline(device_type, use_history, promptTemplate_type=model_type)
    # Interactive questions and answers
    while True:
        
        start_loop = time.time()
        query = input("\nEnter a query: ")
        print('Query entered is : ',query,'\n\n')
        
        if query == "exit":
            break
            
        process_query(query)
        
        
        ### putting the code from process_query
        ### Shouldn't be needing this if process_query runs, but it might fail because of Google search API request  limits 
        
        
        start_query = time.time()
        data = pd.read_csv('SOURCE_DOCUMENTS/image_objects.csv')

        # User query
        # user_query = "Detailed statistics of crimes near Marriott DTLA"
        
    
        print('Is nlp none?? ',nlp == None, 'query none? ',query == None,'\n\n')
        # Function to calculate similarity between the query and detected objects
        def calculate_similarity(query, detected_objects):
            # print('nlp is none ? ',nlp == None, 'query : ',query,'\n\n')
            query_doc = nlp(query)
            detected_doc = nlp(detected_objects)
            return query_doc.similarity(detected_doc)
        
        print('Searching for query : ',query,'\n\n')
        
        
        print('Dataframe data looks like : ',data.shape,'\n\n')
        
        top_related_images = []
        
        if data.shape[0]>0:
            data['similarity'] = data.apply(lambda x: calculate_similarity(query, x['Detected Objects']), axis=1)

            # Sort the DataFrame by similarity in descending order
            sorted_data = data.sort_values(by='similarity', ascending=False)

            # Get the image filenames most related to the user query
            top_related_images = sorted_data['Image Filename'].tolist()

            print('Length of top_related_images : ',len(top_related_images),'\n\n')

            # Display the top related image filenames
            print("Image filenames most related to the user query:")

            print('Values in list : ',top_related_images[:5],'\n\n')
            
            
            
            #we are dumping these images to another folder so that we can show 

            end_query = time.time()

            print('Time taken to find similar images : ',end_query - start_query,'\n\n')
        
        
        #calling ingest for each user query, this takes just 15 seconds with all-miniLM
        
        call(["python3", "ingest.py"])
        
        print('After process_query text...\n\n')
        
        start = time.time()
        # Get the answer from the chain
        res = qa(query)
        answer, docs = res["result"], res["source_documents"]

        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)
        
        end = time.time()
        
        
        
        #display the top-related images here
        
        image_dir = 'images/'
        
        
        
        ### for .py...
        
        # for image_name in top_related_images[:5]:
        #     try:
        #         image_path = os.path.join(image_dir, image_name)
        #         img = Image.open(image_path)
        #         img.show()  
        #     except:
        #         pass
        
        print('Time taken after image is : ',end - start,'\n\n')
        
        
        ### for Colab...
        
#         from IPython.display import Image, display

#         for filename in top_related_images[:5]:
#             try:
#                 display(Image(filename))
#             except:
#                 pass
        
        
        if show_sources:  # this is a flag that you can set to disable showing answers.
            # # Print the relevant sources used for the answer
            print("----------------------------------SOURCE DOCUMENTS---------------------------")
            for document in docs:
                print("\n> " + document.metadata["source"] + ":")
                print(document.page_content)
            print("----------------------------------SOURCE DOCUMENTS---------------------------")
        
        # Log the Q&A to CSV only if save_qa is True
        if save_qa:
            utils.log_to_csv(query, answer)
            
        end_loop = time.time()
        print('Time taken in total is : ',end_loop - start_loop,'\n\n')
        


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s", level=logging.INFO
    )
    main()
