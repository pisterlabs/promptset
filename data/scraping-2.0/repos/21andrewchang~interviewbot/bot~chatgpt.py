import os
import sys
import constants
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
import Levenshtein

os.environ["OPENAI_API_KEY"] = constants.APIKEY

# Get input from command line
user_query = sys.argv[1]

# Search for the closest matching file with a name that matches user_query
matching_file_path = None
min_distance = float('inf')
for file_name in os.listdir("./files/"):
    if file_name.endswith(".txt"):
        distance = Levenshtein.distance(user_query.lower(), file_name.lower())
        if distance < min_distance:
            min_distance = distance
            matching_file_path = os.path.join("./files/", file_name)

if matching_file_path:
    # Read the content of the matching file
    with open(matching_file_path, 'r') as file:
        file_content = file.read()
    
    # Create a TextLoader instance with the filename as document ID
    loader = TextLoader(matching_file_path)
    
    # Create an index using VectorstoreIndexCreator
    index = VectorstoreIndexCreator().from_loaders([loader])
    
    # Provide a system message for the interview
    system_prompt = f"You are an interviewer. You will be given a name and I need you to make 5 customized interview questions based on the resume in '{matching_file_path}' if there isn't a file that matches that exactly then say A file with that name can't be found."
    
    # Combine the system message and user query
    combined_query = f"{system_prompt} {user_query}"
    
    # Query the index
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.8)
    questions = index.query(combined_query, llm)
    
    # Print the generated interview questions
    print(questions)
else:
    print(f"No matching .txt file found in 'files' directory for '{user_query}'.")
