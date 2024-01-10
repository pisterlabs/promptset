from llama_index import ServiceContext, VectorStoreIndex
from llama_index.vector_stores import PineconeVectorStore
from llama_index.llms import OpenAI
import pinecone
import random
import os
import json
import openai
import requests
import keyring
import uuid  # For generating random file names
from concurrent.futures import ThreadPoolExecutor
from llama_index import (
    VectorStoreIndex,
)
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine

index_name = "history-chunks"
total_vector_count = 0

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # This gets the directory of the current script
directory = os.path.join(BASE_DIR, 'output') 
selection_counts_file = directory + '/selection_counts.json'
image_directory = os.path.join(BASE_DIR, 'output') # Assuming 'output' is a sub-directory of the script's directory


import keyring
import os

# Get the API key from the system's keyring
api_key = keyring.get_password("openai", "api_key")

# Check if the API key was retrieved successfully
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
else:
    print("Failed to retrieve the API key.")

pinecone.init(api_key="9e656193-e394-43af-8147-5dcc62a22ef2", environment="asia-southeast1-gcp-free")
openai.api_key = os.environ["OPENAI_API_KEY"]

def connect_to_DB():
    # Use LLamaIndex to break up the text into chunks
    pinecone_index = pinecone.Index(index_name=index_name)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    service_context = ServiceContext.from_defaults(llm=OpenAI())
    return VectorStoreIndex.from_vector_store(vector_store=vector_store, service_context=service_context)


def get_random_chunk():
    # Load and parse the JSON file
    with open(directory+"/nodes.json", 'r') as file:
        nodes = json.load(file)
    
    # Load selection counts or initialize with zeros if it doesn't exist
    try:
        with open(selection_counts_file, 'r') as file:
            selection_counts = json.load(file)
    except FileNotFoundError:
        selection_counts = {node['id_']: 0 for node in nodes}
    
    # Calculate weights
    weights = [1/(selection_counts.get(node['id_'], 0) + 1) for node in nodes]
    
    # Weighted random choice
    random_node = random.choices(nodes, weights, k=1)[0]
    
    # Update and save the selection count
    selection_counts[random_node['id_']] = selection_counts.get(random_node['id_'], 0) + 1
    with open(selection_counts_file, 'w') as file:
        json.dump(selection_counts, file)
    
    # Extract and return the text of the randomly selected node
    return random_node

def check_answer(user_answer, correct_answer):
    # Implement your answer checking logic here
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"User Answer:{user_answer}\n\nCorrect Answer:{correct_answer}\n\nRespond in a very short way - is the User Answer when compared to Correct Answer correct or partially correct or incorrect? In case User Answer is empty or makes no sense it is incorrect"},
                ]
            )
                # Extract and store response
        print(response['choices'][0]['message']['content'])

    except Exception as e:
        print(f"Error processing text: {response.choices[0].text.strip()}. Error: {str(e)}")

    print("The correct answer: " + correct_answer.lower())
    return response['choices'][0]['message']['content']

def get_all_chunks_text(directory):
    # Load and parse the JSON file
    with open(directory + "/nodes.json", 'r') as file:
        nodes = json.load(file)
    
    # Collect the text of each node into a list
    all_chunks_text = [node['text'] for node in nodes]
    
    # Return the list of all chunks text
    return all_chunks_text


def Test():
    text = get_random_chunk()['text']
    question_answer = None

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{text}\n\nGiven the text come-up with a question and answer pair. Must format the response as a JSON object with 'question' and 'answer' fields."},
                ]
            )
                # Extract and store response
        question_answer = response['choices'][0]['message']['content']  # Assume each bullet point is on a new line

    except Exception as e:
        print(f"Error processing text: {text}. Error: {str(e)}")
        question_answer = "Error generating description"

    # Parse the question and answer from the API response
    if question_answer:
        try:
            qa_json = json.loads(question_answer)
            question = qa_json.get('question', '')
            correct_answer = qa_json.get('answer', '')
            
            # Print the question part
            print(f"Question: {question}")
            
            # Wait for the human to provide an answer
            user_answer = input("Your answer: ")
            
            # Check the provided answer
            check_answer(user_answer, correct_answer)
            
        except json.JSONDecodeError:
            print(f"Error parsing the question and answer JSON: {question_answer}")

    else:
        print("No question and answer generated.")

def get_question(text):
    question_answer = None
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"{text}\n\nGiven the text come-up with a question and answer pair. Must format the response as a JSON object with 'question' and 'answer' fields."},
                ]
            )
                # Extract and store response
        question_answer = response['choices'][0]['message']['content']  # Assume each bullet point is on a new line

    except Exception as e:
        print(f"Error processing text: {text}. Error: {str(e)}")
        question_answer = "Error generating description"

    return question_answer


def Question_and_Image():
    random_chunk = get_random_chunk()
    text = random_chunk['text']
    description = None
    question_answer = None
    json_question_answer = None

    with ThreadPoolExecutor() as executor:
        futures = {
            'Image': executor.submit(Image, text),
            'get_question': executor.submit(get_question, text)
        }

    results = [(fn, future.result()) for fn, future in futures.items()]

    for function_name, response in results:
        if function_name == "Image":
            description_and_location_json = response
        elif function_name == "get_question":
            question_answer = response

    # Parse the question and answer from the API response
    if question_answer:
        try:
            qa_json = json.loads(question_answer)
            question = qa_json.get('question', '')
            answer = qa_json.get('answer', '')
           
            description_and_location = json.loads(description_and_location_json)
            description = description_and_location.get('description','')
            file_location = description_and_location.get('file_location','')
            
            # Convert to dictionary
            data = {
                "question": question,
                "answer": answer,
                "description": description,
                "file_location":file_location,
                "chunk": text,
                "page": random_chunk['page_number'],
                "pdf_file": random_chunk['document_name'][:-3] + 'pdf'
            }

            for key, value in data.items():
                if key != "chunk":
                    print(f"{key}: {value}")

            # Generate a random string for the filename
            #randomness = str(uuid.uuid4())[:8]  # You can adjust the length as needed
            filename = f"QA_{ data.get('file_location','') }.JSON"

            # Write data to the JSON file
            with open(image_directory + "/" + filename, 'w') as json_file:
                json.dump(data, json_file, indent=4)

            # Convert dictionary to JSON string
            json_question_answer = json.dumps(data)
            
        except json.JSONDecodeError:
            print(f"Error parsing the question and answer JSON: {question_answer}")

    else:
        print("No question and answer generated.")

    return json_question_answer


def Answer(index, question):
    query_engine = index.as_query_engine(
        similarity_top_k=8, 
        response_mode='refine',
        verbose=True)
    response = query_engine.query(question)
    print(response)

def Image(text=None):
    image_description = None

    if text == None:
        text = get_random_chunk()['text']

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"7 word or less, describe a single physical object or scene that is mentioned in this text. Those must be easy to draw. Here is the text: {text}\n\n"},
                ]
            )
                # Extract
                #  and store response
        image_description = response['choices'][0]['message']['content']  # Assume each bullet point is on a new line

    except Exception as e:
        print(f"Error processing text: {text}. Error: {str(e)}")
        image_description = "Error generating description"

    description = image_description
    response = openai.Image.create(
        prompt=image_description,
        n=1,
        size="512x512"
    )
    image_url = response['data'][0]['url']

    # Generate a random name for the image
    random_filename = f"{uuid.uuid4()}.jpg"
    file_location = os.path.join(image_directory, random_filename)

    # Send a GET request to the image URL
    response = requests.get(image_url)

    # Check if the request was successful
    if response.status_code == 200:
        # Specify the local path where the image will be saved
        with open(file_location, "wb") as file:
            file.write(response.content)
    else:
        print("Failed to fetch the image.")

    # Return a JSON response containing the description and file_location
    return json.dumps({"description": description, "file_location": random_filename})

def Summarize():
     # Open and immediately close 'Summary.txt' in write mode to clear its contents
    with open(directory + '/Summary.txt', 'w') as file:
        pass
    all_chunks_text = get_all_chunks_text(directory)
    chunk_count = 1
    import time

    for chunk in all_chunks_text:
        success = False  # Variable to track if processing the chunk succeeded
        retries = 0  # Variable to track the number of retries

        chunk_count += 1

        while not success and retries < 3:  # Assuming a maximum of 3 retries per chunk
            try:
                # Make API call
                response = openai.ChatCompletion.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": f"{chunk}\n\nProvide bullet points only of the main hisotrical insights and key facts, such as names, locations, dates, numbers. Do not include citations. Must be no more than 6 points, but can be less if no important information. Avoid duplication of the same facts"},
                    ]
                )
                # Extract and store response
                bullet_points = response['choices'][0]['message']['content']  # Assume each bullet point is on a new line
                
                print(f"processing chunk {chunk_count}")
                
                # Append the bullet_text to the file called 'Summary.txt'
                with open(directory + '/Summary.txt', 'a') as file:
                    file.write(bullet_points + '\n')  # Add a newline character at the end for separation
                
                success = True  # Set success to True if the above code executes without throwing an exception

            except Exception as e:
                print(f"Error processing text chunk: {chunk}. Error: {str(e)}. Retrying in 5 seconds...")
                retries += 1  # Increment the retries count
                time.sleep(5)  # Wait for 5 seconds before retrying

        if not success:
            print(f"Failed to process chunk {chunk} after {retries} retries.")

    # Optionally, you can still return all_bullet_points if needed
    with open(directory + '/Summary.txt', 'r') as file:
        all_bullet_points = file.read().splitlines()
    
    return all_bullet_points  # Return the concatenated list of all bullet points


def get_chunk_count(index):
    # Fetch all ids from the Pinecone index
    index = pinecone.Index(index_name)
    global total_vector_count
    total_vector_count = index.describe_index_stats()['total_vector_count']
    print(total_vector_count)

def main():
    index = connect_to_DB()
    get_chunk_count(index)

    while True:
        print("\nMenu:")
        print("Test - T")
        print("Image - I")
        print("Answer - A")
        print("Summarize - S")
        print("Quit - Q")
        choice = input("Enter your choice: ").upper()

        if choice == 'T':
            Test()
        elif choice == 'S':
            Summarize()
        elif choice == 'A':
            question = input("Enter your question: ")
            Answer(index, question)
        elif choice == 'I':
            Image()
        elif choice == 'Q':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
