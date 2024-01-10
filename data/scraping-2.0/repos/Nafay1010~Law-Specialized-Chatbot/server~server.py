from flask import Flask, jsonify, request, make_response
from flask_cors import CORS
from llama_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper, ServiceContext
from langchain.chat_models import ChatOpenAI
import json
import os
import re
import PyPDF2
from youtube_transcript_api import YouTubeTranscriptApi
import wolframalpha

global index


app = Flask(__name__)
CORS(app)
app.secret_key = ""
# OpenAI API key
os.environ["OPENAI_API_KEY"] = ""
# Create LLMPredictor and the index once
llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo", max_tokens=2000))
index = None  # Initialize to None, it will be constructed later
empty_chat_history = []
# Open the file in write mode and write the empty list as JSON
with open('context_data/chat_history.json', 'w') as f:
    json.dump(empty_chat_history, f, indent=4)


def construct_index(directory_path):
    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 2000
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600 

    # define prompt helper
    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    # define LLM
    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.5, model_name="gpt-3.5-turbo", max_tokens=num_outputs))
 
    documents = SimpleDirectoryReader(directory_path).load_data()
    
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)

    index.save_to_disk('index.json')

    return index

def save_chat_history(filename, chat_history):
    with open(filename, 'w') as f:
        json.dump(chat_history, f, indent=4)

def load_chat_history(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            chat_history = json.load(f)
        return chat_history
    else:
        return []

        
def handle_user_input(user_input, message_history):
    # Add the user input to the message history
    index = GPTSimpleVectorIndex.load_from_disk('index.json')

    message_history.append({"role": "user", "content": user_input})
    
    # Combine the entire message history for context
    context = " ".join(item["content"] for item in message_history)
    
    # Generate a response with context
    response = index.query(context, response_mode='compact')
    
    # Add the system response to the message history
    message_history.append({"role": "system", "content": response.response.strip()})
    
    return response.response.strip()

index = construct_index("context_data/data")
index = GPTSimpleVectorIndex.load_from_disk('index.json')

@app.route("/chatbot", methods=['POST'])
def get_response():
    data = request.get_json()
    user_input = data[1].get('latestText')  # User's prompt
    message_history = load_chat_history("context_data/chat_history.json")  # Load chat history
    response = handle_user_input(user_input, message_history)
    save_chat_history("context_data/chat_history.json", message_history)
    source=handle_user_input('Give me the link or source of information of'+ user_input, message_history)
    print(source)
    return jsonify(response, source)

@app.route('/updateContext', methods=['POST'])
def update_context():
    data = request.get_json()
    print(data)
    if data is not None:
        context = json.loads(data.get('context'))
        processed_context = [
            {key: value for key, value in message.items() if key != '_id'}
            for message in context
        ]
        print(processed_context)
        if processed_context is not None:
            save_chat_history("context_data/chat_history.json", processed_context)
            print('Context Updated')
            # construct_index()
            return jsonify("Context Updated")
        else:
            print('Context is None')
            return jsonify("Context is None")
    else:
        print('Context is None')
        return jsonify("Context is None")

@app.route('/uploadpdf', methods=['POST'])
def upload_pdf():
    file = request.files['file']
    name = request.form['name']
    source = request.form['source']

    if file:
        result = pdf_to_txt(file, "context_data/data/PDFs", name, source)
        if result:
            # construct_index("context_data/data")
            return jsonify({"message": "File uploaded and converted successfully", "text_file": result})
        else:
            return jsonify({"error": "Failed to convert the PDF"})
    else:
        return jsonify({"error": "No file provided in the request"})

def pdf_to_txt(input_pdf_file, output_dir=None, name=None, source=None):
    try:
        # Ensure the output directory exists
        if output_dir is None:
            output_dir = os.path.dirname(input_pdf_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(input_pdf_file)

        # Initialize an empty string to store text
        text = ""
        formulas = []

        # Loop through each page and extract text
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            text += page_text

            # Find all formulas in the current page text and strip whitespace
            formulas_on_page = re.findall(r'<formula>(.*?)</formula>', page_text, re.DOTALL)
            formulas.extend([formula.replace(" ", "").strip() for formula in formulas_on_page])

        # Append name and source information to the end of the text content
        text += f"\nSource of Information: {source}"

        # Generate the output TXT file name based on the input file name
        output_txt_file = os.path.join(output_dir, os.path.splitext(os.path.basename(name))[0] + '.txt')

        # Write the extracted text to the output TXT file
        with open(output_txt_file, 'a', encoding='utf-8') as txt_output:
            txt_output.write(text)

        # Write the extracted formulas to the formula TXT file
        formula_txt_file = os.path.join('context_data/data', 'formulas.txt')
        with open(formula_txt_file, 'a', encoding='utf-8') as formula_output:
            for formula in formulas:
                formula_output.write(formula + '\n')

        print(f"PDF converted to TXT: {output_txt_file}")
        print(f"Formulas extracted to TXT: {formula_txt_file}")
        return output_txt_file
    except Exception as e:
        print(f"Error converting PDF to TXT: {e}")
        return None

@app.route('/uploadjson', methods=['POST'])
def upload_json():
    file = request.files['file']
    name = request.form['name']
    source = request.form['source']
    print(file, name, source)

    if file:
        temp_json_path = save_uploaded_json(file, "context_data/data/JSON", name)
        if temp_json_path:
            result = json_to_txt(temp_json_path, "context_data/data/JSON", name, source)
            # construct_index("context_data/data")
            return jsonify({"message": "File uploaded and converted successfully", "text_file": result})
        else:
            return jsonify({"error": "Failed to convert the JSON"})
    else:
        return jsonify({"error": "No file provided in the request"})

def save_uploaded_json(uploaded_file, output_dir, name):
    try:
        # Ensure the output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Generate the temporary JSON file path
        temp_json_path = os.path.join(output_dir, f"temp_{name}.json")

        # Save the uploaded JSON file to the temporary path
        uploaded_file.save(temp_json_path)

        return temp_json_path
    except Exception as e:
        print(f"Error saving uploaded JSON: {e}")
        return None

def json_to_txt(input_json_file, output_dir, name, source):
    try:
        # Read the input JSON file
        with open(input_json_file, 'r', encoding='utf-8') as json_input:
            data = json.load(json_input)

        # Convert the JSON data to a formatted string
        text = json.dumps(data, indent=4)

        # Append name and source information to the end of the text content
        text += f"\nSource of Information: {source}"

        # Generate the output TXT file path in the specified directory
        output_txt_file = os.path.join(output_dir, os.path.splitext(name)[0] + '.txt')

        # Write the formatted JSON text with name and source information to the output TXT file
        with open(output_txt_file, 'w', encoding='utf-8') as txt_output:
            txt_output.write(text)

        print(f"JSON converted to TXT: {output_txt_file}")

        # Delete the temporary JSON file
        os.remove(input_json_file)

        return output_txt_file
    except Exception as e:
        print(f"Error converting JSON to TXT: {e}")
        return None

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.get_json()
    #open file /context_data/data/prompt.txt in append mode and write the data
    with open('context_data/data/feedback.txt', 'a') as f:
        for i in data:
            f.write(i.strip() + '\n')
    construct_index("context_data/data")
    return jsonify("Feedback Trained Successfully")

@app.route('/getFormulas', methods=['GET'])  # Renamed for consistency
def get_formulas():
    try:
        with open('context_data/data/formulas.txt', 'r', encoding='utf-8') as f:
            data = []
            for line in f.readlines():
                # Strip the line of whitespace and newline characters
                formula = line.strip()
                # Find all parameters within curly braces
                parameters = re.findall(r'{(.*?)}', formula)
                # Remove the curly braces to clean the formula
                clean_formula = re.sub(r'{(.*?)}', r'\1', formula)
                # Append both the clean formula and parameters to the data list
                data.append({'formula': clean_formula, 'parameters': parameters})
        response = make_response(jsonify(data))
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        return response
    except FileNotFoundError:
        return jsonify({"error": "Formulas file not found."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/calculateFormula', methods=['POST'])
def calculate_formula():
    data = request.get_json()
    print(data)
    formula = data.get('formula')
    parameters = data.get('parameters')
    result = compute_formula_with_wolfram(formula, **parameters)
    return jsonify(result)
# Function to compute a mathematical problem using Wolfram Alpha
def compute_formula_with_wolfram(formula, **parameters):
    # Your Wolfram Alpha App ID
    app_id = '44X2XG-T5K5ULARUK'
    client = wolframalpha.Client(app_id)

    # Construct the query by substituting the parameters into the formula
    query = formula
    for param, value in parameters.items():
        query = query.replace(str(param), str(value))

    try:
        # Making the query to Wolfram Alpha
        res = client.query(query)

        # Initialize an empty list to collect the plaintext results
        results = []

        # Processing the response
        for pod in res.pods:
            for sub in pod.subpods:
                if hasattr(sub, 'plaintext') and sub.plaintext:
                    results.append(sub.plaintext)

        # Join all the results into a single string
        response_string = '\n'.join(results)
        return response_string

    except Exception as e:
        return f"An error occurred: {e}"


@app.route('/addFormula', methods=['POST'])
def add_formula():
    data = request.get_json()
    if data is not None:
        print("formula", data)
        formula = data.get('formula')
        with open('context_data/data/formulas.txt', 'a', encoding='utf-8') as f:
            f.write('\n' + formula)
        return jsonify("Formula added successfully"), 200
    else:
        return jsonify("Formula not added"), 400


@app.route('/prompt', methods=['POST'])
def training_by_prompt():
    data = request.get_json()
    question = data.get('question')
    answer = data.get('answer')
    url = data.get('url')
    with open('context_data/data/prompts.txt', 'a') as f:
        f.write(question + '\n' + answer + '\n' + 'Source (Link) of information: ' + url + '\n')
    construct_index("context_data/data")
    return jsonify('Prompt Training Successfully')

@app.route("/trash", methods=['GET'])
def clear_conversation():
    # Create an empty list
    empty_chat_history = []
    # Open the file in write mode and write the empty list as JSON
    with open('context_data/chat_history.json', 'w') as f:
        json.dump(empty_chat_history, f, indent=4)
    return jsonify("Conversation cleared")

#VidLink Logic
def create_link_text(id, url):
    outl = []
    lang = ['en', 'es','sp']
    text = YouTubeTranscriptApi.get_transcript(id, lang)

    for i in text:
        outx = (i['text'])
        outl.append(outx)

        with open("context_data/data/video_transcript.txt", "a") as opf:
            opf.write(outx + "\n")
    
    with open('context_data/data/video_transcript.txt', 'a') as f:
        f.write("Source (Link) of Information: " + str(url) + "\n")


def get_vid_id(url):
    if "https://www.youtube.com/watch?v=" in url:
        # Find the index where the video ID starts
        start_index = url.index("https://www.youtube.com/watch?v=") + len("https://www.youtube.com/watch?v=")

        # Extract the video ID
        video_id = url[start_index:start_index + 11]

        if video_id.startswith('-'):
            video_id = '\\' + video_id

        return video_id
    else:
        return None
    

@app.route('/vidURL', methods=['POST'])
def vidURL():
    data = request.get_json()
    id = get_vid_id(data)
    if(id):
        create_link_text(id, data)
        construct_index("context_data/data")
        return jsonify("Successfully Transcribed")
    else:
        return jsonify("Video ID not found.")


if __name__ == '__main__':
    app.run(debug=True, port=8080)
