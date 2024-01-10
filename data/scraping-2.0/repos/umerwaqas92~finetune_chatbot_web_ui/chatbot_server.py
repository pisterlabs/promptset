from flask import Flask, request, jsonify, render_template, redirect
import requests
import json
import os
from llama_index import SimpleDirectoryReader, GPTListIndex, readers, GPTSimpleVectorIndex, LLMPredictor, PromptHelper,ServiceContext
from langchain import OpenAI
from flask_cors import CORS
import datetime
from PyPDF2 import PdfReader
import logging
from werkzeug.serving import make_server


log_file = 'error.log'


app = Flask(__name__)
CORS(app)

# Add an error handler
@app.errorhandler(Exception)
def handle_exception(e):
    # Log the exception
    app.logger.exception(e)
    
    # Return a JSON error response
    response = jsonify({'error': str(e)})
    response.status_code = 500
    return response




# os.environ["OPENAI_API_KEY"]="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"

my_dir = os.path.dirname(__file__)
api_key_json = os.path.join(my_dir, 'api_key.json')
DATA_FILE = os.path.join(my_dir, 'query_responses.json')

islogin =False
password = "123456"
username = "admin"



#whatsapp
VERIFICATION_TOKEN = '123456789'
ACCESS_TOKEN = 'EAAwBWAci3sIBAJjc9s9Xvh6EqRsZBNke6nSTNBDrwTp5dgZCV18NbRc6dtUJFQRPZA6UXiZC4xPYtD75ZBZAg8CttANHCvSDITCttXglZBKYyTIMAw4alGUJRAQB1qqsnNMQEY5SXRCZB04Nf9iUasQmmhnU4c7S7bZAFBa0eiFLpyxYGTRW8jy7CxpwEADhhgNgZB4SThR0AnggZDZD'
Phone_Number_ID = "102566059470141"


def update_config():
    with open(api_key_json, 'r') as f:
        jsonfile= json.load(f)
        api_key = jsonfile['api_key'].strip()
        print(jsonfile)
        os.environ["OPENAI_API_KEY"]=api_key.strip()
        print("set api key |"+os.environ["OPENAI_API_KEY"]+"|")
        print("set api key |"+"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"+"|")
        api_temp = jsonfile['api_temp']
        api_model_name = jsonfile['api_model_name']
        api_token_max = jsonfile['api_token_max']
        
 
        


update_config()


# Route to handle subscription verification requests
@app.route('/webhook', methods=['GET'])
def handle_verification():
    # Verify the subscription request
    if request.args.get('hub.verify_token') == VERIFICATION_TOKEN:
        return request.args.get('hub.challenge')
    else:
        return 'Invalid verification token'

@app.route('/webhook', methods=['GET', 'POST'])
def webhook():
    if request.method == 'GET':
        if request.args.get('hub.mode') == 'subscribe' and request.args.get('hub.verify_token') == VERIFICATION_TOKEN:
            return request.args.get('hub.challenge')
        else:
            return 'Invalid verification token'
    elif request.method == 'POST':
        data = request.json
        message = data['entry'][0]['changes'][0]['value']['messages'][0]['text']['body']
        print('Incoming webhook: ' + message)
        sender = data['entry'][0]['changes'][0]['value']['messages'][0]['from']
        recipient = data['entry'][0]['changes'][0]['value']['metadata']['display_phone_number']
        response=ask_ai3(message)
        send_message(sender, response)
        return 'Message received'
    
def send_message(recipient_phone_number, message):
    version = "v16.0"
    url = f"https://graph.facebook.com/{version}/{Phone_Number_ID}/messages"

    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": recipient_phone_number,
        "type": "text",
        "text": {"body": message}
    }

    headers = {
        'Authorization': f'Bearer {ACCESS_TOKEN}',
        'Content-Type': 'application/json'
    }

    response = requests.post(url, headers=headers, json=payload)
    print("Sent message")
    print(response.text)


    




def extract_text_from_pdfs(path_to_pdf):
    with open(path_to_pdf, 'rb') as f:
        pdf_reader = PdfReader(f)
        text = ''
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# def construct_index( api_temp, api_model_name, api_token_max):
    

#     # set maximum input size
#     max_input_size = 4096
#     # set number of output tokens
#     num_outputs = int(api_token_max)
#     # set maximum chunk overlap
#     max_chunk_overlap = 50
#     # set chunk size limit
#     chunk_size_limit = 1195

#     my_dir = os.path.dirname(__file__)
#     pickle_file_path = os.path.join(my_dir, 'context_data/data')


#     # define LLM

#     llm_predictor = LLMPredictor(llm=OpenAI(temperature=api_temp, model_name=api_model_name, max_tokens=num_outputs))
#     prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

#     documents = SimpleDirectoryReader(pickle_file_path).load_data()

#     index = GPTSimpleVectorIndex(
#         documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper
#     )

#     index.save_to_disk(os.path.join(my_dir, 'index.json'))

#     return index


my_dir = os.path.dirname(__file__)
pickle_file_path = os.path.join(my_dir, 'index.json')
index2 = GPTSimpleVectorIndex.load_from_disk(pickle_file_path)



def ask_ai(query):
    # update_config()
    # if(index2==None):
    #     pickle_file_path = os.path.join(my_dir, 'index.json')
    # print(os.environ["OPENAI_API_KEY"])
    # os.environ["OPENAI_API_KEY"] = "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"

   
    response = index2.query(query, response_mode="compact")
    return {"response": response.response}



def construct_index(api_key, api_temp, api_model_name, api_token_max):
    # set api key
    # os.environ["OPENAI_API_KEY"]=api_key

    # set maximum input size
    max_input_size = 4096
    # set number of output tokens
    num_outputs = 2000
    # set maximum chunk overlap
    max_chunk_overlap = 20
    # set chunk size limit
    chunk_size_limit = 600 

    # define LLM
    
    my_dir = os.path.dirname(__file__)
    pickle_file_path = os.path.join(my_dir, 'index.json')

    
    # llm_predictor = LLMPredictor(llm=OpenAI(temperature=float(api_temp), model_name=api_model_name, max_tokens=int(api_token_max)))
    # prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)
 
    # documents = SimpleDirectoryReader(pickle_file_path).load_data()
    
    # index = GPTSimpleVectorIndex(
    #     documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper
    # )

    # index.save_to_disk('index.json')

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    # define LLM
    llm_predictor = LLMPredictor(llm=OpenAI(temperature=float(api_temp), model_name=api_model_name, max_tokens=int(api_token_max)))
 
    documents = SimpleDirectoryReader(my_dir+"/context_data/data").load_data()
    
    service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)
    index = GPTSimpleVectorIndex.from_documents(documents, service_context=service_context)

    index.save_to_disk('index.json')
    global index2

    index2=index

    return index


@app.route('/restart', methods=['Get', 'POST'])
def restart():
    # Check for a restart token in the request body
    # if request.form.get('token') == 'my_restart_token':
        # Close the server socket
    shutdown_func = request.environ.get('werkzeug.server.shutdown')
    if shutdown_func is not None:
        shutdown_func()

    # Restart the server
    make_server('localhost', 5000, app).serve_forever()
    redirect('/')


    # return 'Server restarted'



@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # get data from json file   
        if request.form['password'] == password and request.form['email'] == username:
            islogin=True
            return redirect('/')
        else:
            render_template('login.html', error="Invalid username or password")
    return render_template('login.html')

@app.route('/clearchhat')
def clear_chat_history():
    with open(DATA_FILE, 'w') as f:
            json.dump({"":"\nHi there! feel free to ask!"}, f)
    return redirect('/')


@app.route('/answer', methods=['POST'])
def answer():
    query = request.json['query']
    
    try:
        response = ask_ai(query)
    except Exception as e:
        response = {"response": "Sorry, I'm not sure how to answer that."}
    return jsonify(response)

@app.route('/save', methods=['POST'])
def save():
    content = request.form['content']
    filename = request.form['filename']

    filepath = os.path.join(my_dir+'/context_data/data', filename,)
    with open(filepath, 'w') as f:
        f.write(content)
            
    return redirect('/edit?filename=' + filename)


@app.route('/delete', methods=['POST'])
def delete():
    filename = request.json['filename']
    filepath = os.path.join(my_dir+'/context_data/data', filename)
    os.remove(filepath)

    return 'OK'



# Load query-response data from file, or create empty dict if file does not exist
if os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'r') as f:
        query_responses = json.load(f)
else:
    query_responses = {}





# def ask_ai(query):
#     my_dir = os.path.dirname(__file__)

#     pickle_file_path = os.path.join(my_dir, 'index.json')
#     # set api key
#     # os.environ["OPENAI_API_KEY"]="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"
#     index = GPTSimpleVectorIndex.load_from_disk(pickle_file_path)

#     response = index.query(query, response_mode="compact")
#     return {"response": response.response}


@app.route('/chatbot')
def chatbot():
    return render_template('/chatbot.html')




@app.route('/create_new_page', methods=['POST'])
def create_new_page():
    # Get form data
    
    filename_input = request.form.get('filename')
    
    filepath = os.path.join(my_dir+'/context_data/data',  f"{filename_input}.txt")
    
   
    with open(filepath, 'w') as f:
        f.write('')
    
    # Return success response
    return redirect('/edit?filename=' +  f"{filename_input}.txt")





@app.route('/edit')
def edit():
    filename = request.args.get('filename')
    filepath = os.path.join(my_dir+'/context_data/data', filename)
    print(filepath)
    with open(filepath, 'r') as f:
        content = f.read()
    return render_template('editor.html', content=content, filename=filename)



def get_files():
    list_dir=os.path.join(my_dir, 'context_data/data')
    files = os.listdir(list_dir)
    return files


@app.route('/files')
def file_list():
    files = get_files()
    # Render the file list template with the list of files
    return render_template('file_list.html', files=files)


@app.route('/', methods=['GET', 'POST'])
def index():
    # Load query-response data from file, or create empty dict if file does not exist
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            query_responses = json.load(f)
    else:
        query_responses = {}


    query_responses = {k: v for k, v in reversed(query_responses.items())}


    if request.method == 'POST':
        query = request.form['query']

        try:
        # Get response from AI
            response = ask_ai(query)
            response_text = response['response']
        except:
            response_text = "Sorry, I don't understand."

        # Save query-response pair to dictionary and write to file
        query_responses[query] = response_text
        query_responses = {k: v for k, v in reversed(query_responses.items())}

        with open(DATA_FILE, 'w') as f:
            json.dump(query_responses, f)

        return render_template('index.html', query_responses=query_responses)
    return render_template('index.html', query_responses=query_responses)


# @app.route('/uploadFile', methods=['GET', 'POST'])
# def upload_file():
#     file_list = get_files()
#     if request.method == 'POST':
#         document_files = request.files.getlist('documents')
#         print("got files "+str(document_files))
#         print(len(document_files))
#         if len(document_files) > 0:
#             for document_file in document_files:
#                 document_filename = document_file.filename
#                 if document_filename:
#                     document_path = os.path.join(my_dir+'/context_data', 'data', document_filename,)
#                     document_file.save(document_path)
#             file_list = get_files() # assuming this function returns a list of file paths
#             return render_template('upload.html', success='File(s) uploaded successfully', api_key=api_key, api_temp=api_temp, api_model_name=api_model_name, api_token_max=api_token_max, file_list=file_list)
#         else:
#             return render_template('upload.html', error='No file selected', api_key=api_key, api_temp=api_temp, api_model_name=api_model_name, api_token_max=api_token_max, file_list=file_list)
#     else:
#         return render_template('upload.html', api_key=api_key, api_temp=api_temp, api_model_name=api_model_name, api_token_max=api_token_max, file_list=file_list)


@app.route('/uploadFile', methods=['GET', 'POST'])
def uploadFile():
    
     document_file = request.files['document']

    
     file_list=get_files()
     print("got files "+str(len(file_list)))

     document_filename = document_file.filename
     if document_filename:
        document_path = os.path.join(my_dir+'/context_data', 'data', document_filename,)
        #ceck if file isd pdf
        
        document_file.save(document_path)

        if document_filename.lower().endswith('.pdf') :
            print("pdf file")
            #convert pdf to txt
            text = extract_text_from_pdfs(document_path)
            #save txt file
            document_filename_new = document_filename.lower().replace(".pdf", ".txt")
            document_path_new = os.path.join(my_dir+'/context_data', 'data', document_filename_new,)
            print("saving file "+document_path_new)
            with open(document_path_new, 'w') as f:
                f.write(text)
                os.remove(document_path)
           

        
        file_list=get_files()
      
        return  redirect("/upload") #render_template('upload.html',success='File uploaded successfully',api_key=api_key, api_temp=api_temp, api_model_name=api_model_name, api_token_max=api_token_max,file_list=file_list)
     
     return redirect("/upload")  #render_template('upload.html',error='File is not selected',api_key=api_key, api_temp=api_temp, api_model_name=api_model_name, api_token_max=api_token_max, file_list=file_list)


def ask_ai2(query):
    my_dir = os.path.dirname(__file__)
    pickle_file_path = os.path.join(my_dir, 'islamic_index.json')

    index = GPTSimpleVectorIndex.load_from_disk(pickle_file_path)

    response = index.query(query, response_mode="compact")

    return {"response": response.response}

def ask_ai3(query):
    my_dir = os.path.dirname(__file__)
    pickle_file_path = os.path.join(my_dir, 'index.json')
    index = GPTSimpleVectorIndex.load_from_disk(pickle_file_path)
    response = index.query(query, response_mode="compact")
        # Save log data to chat_log.txt
    log_path = os.path.join(my_dir, 'chat_log.txt')
    with open(log_path, 'a') as log_file:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_file.write(f'{timestamp}: {query} --> {response.response}\n')

    return response.response


@app.route('/show_log')
def show_log_api():
    my_dir = os.path.dirname(__file__)
    log_path = os.path.join(my_dir, 'chat_log.txt')

    if not os.path.exists(log_path):
        return {'error': 'Log file not found.'}

    with open(log_path, 'r') as log_file:
        log_data = log_file.read()

    return {'log_data': log_data}


@app.route('/answer3', methods=['POST'])
def answer3():
    query = request.json['query']
    response = ask_ai3(query)
    return response



@app.route('/upload', methods=['GET', 'POST'])
def upload():
    files = get_files()
    if(login == False):
        return redirect("/login")
    if request.method == 'POST':
        
        if 'upload-file"' in request.form:
            #save file
            file = request.files['file']
            filename = secure_filename(file.filename)
            file.save(os.path.join(my_dir+'/context_data/data', filename))


            return render_template('upload.html',success='File uploaded successfully',api_key=api_key, api_temp=api_temp, api_model_name=api_model_name, api_token_max=api_token_max,files=files)
        if 'save_config' in request.form:
            api_key = request.form['api_key'].strip()
            api_temp = request.form['api_temp'].strip()
            api_model_name = request.form['api_model_name'].strip()
            api_token_max = request.form['api_token_max'].strip()
            
            os.environ['OPENAI_API_KEY'] = api_key
            with open(my_dir+'/api_key.json', 'w') as f:
                json.dump({'api_key':api_key,'api_temp':api_temp,'api_model_name':api_model_name,'api_token_max':api_token_max}, f)
                return render_template('upload.html', success='Saved Configuratin',api_key=api_key, api_temp=api_temp, api_model_name=api_model_name, api_token_max=api_token_max,files=files)
        if 'chat_screen' in request.form:
            return render_template('index.html')

        if 'start-training' in request.form:

            api_key = request.form['api_key'].strip()
            api_temp = request.form['api_temp'].strip()
            api_model_name = request.form['api_model_name'].strip()
            api_token_max = request.form['api_token_max'].strip()
           
            os.environ['OPENAI_API_KEY'] = api_key
            with open(my_dir+'/api_key.json', 'w') as f:
                json.dump({'api_key':api_key,'api_temp':api_temp,'api_model_name':api_model_name,'api_token_max':api_token_max}, f)
                try:
                    construct_index(api_key=api_key, api_temp=api_temp,api_model_name=api_model_name,api_token_max=api_token_max)
                except Exception as e:
                    print(e)
                    return render_template('upload.html', error='Training failed, make sure your payment is added in at https://platform.openai.com/account/billing/overview',api_key=api_key, api_temp=api_temp, api_model_name=api_model_name, api_token_max=api_token_max,files=files)
                
                return render_template('upload.html', success='Trained successfully',api_key=api_key, api_temp=api_temp, api_model_name=api_model_name, api_token_max=api_token_max,files=files)

        if'submit' in request.form:
            # Check if the file is present in the request
            if 'document' not in request.files:

                with open(my_dir+'/api_key.json', 'r') as f:
                                jsonfile= json.load(f)
                                api_key = jsonfile['api_key'].strip()
                                api_temp = jsonfile['api_temp']
                                api_model_name = jsonfile['api_model_name']
                                api_token_max = jsonfile['api_token_max']
                                return render_template('upload.html', success='configuration has been saved',api_key=api_key, api_temp=api_temp, api_model_name=api_model_name, api_token_max=api_token_max,files=files)
            else:
                # Get the file from the request and save it to the data directory
                document_file = request.files['document']
                document_filename = document_file.filename
                if not document_filename:

                    api_key = request.form['api_key'].strip()
                    api_temp = request.form['api_temp'].strip()
                    api_model_name = request.form['api_model_name'].strip()
                    api_token_max = request.form['api_token_max'].strip()


                    with open(api_key_json, 'w') as f:
                        json.dump({'api_key': api_key,"api_temp":api_temp,"api_model_name":api_model_name,"api_token_max":api_token_max}, f)

                    return render_template('upload.html', error='No file selected but api has been uploaded',api_key=api_key, api_temp=api_temp, api_model_name=api_model_name, api_token_max=api_token_max,files=files)

                document_path = os.path.join(my_dir+'/context_data', 'data', document_filename,)
                document_file.save(document_path)

            # Update the OpenAI API key in api_key.json
            #filepath = os.path.join(my_dir+'/context_data/data', filename)
            api_key = request.form['api_key']
            with open(my_dir+'/api_key.json', 'w') as f:
                json.dump({'api_key': api_key}, f)

            return render_template('upload.html', success='File uploaded successfully',api_key=api_key, api_temp=api_temp, api_model_name=api_model_name, api_token_max=api_token_max,files=files)
        else:
            with open(my_dir+'/api_key.json', 'r') as f:
                jsonfile= json.load(f)
                api_key = jsonfile['api_key'].strip()
                api_temp = jsonfile['api_temp']
                api_model_name = jsonfile['api_model_name']
                api_token_max = jsonfile['api_token_max']

                return render_template('upload.html',api_key=api_key, api_temp=api_temp, api_model_name=api_model_name, api_token_max=api_token_max,files=files)
    else:
         with open(my_dir+'/api_key.json', 'r') as f:
                jsonfile= json.load(f)
                api_key = jsonfile['api_key'].strip()
                api_temp = jsonfile['api_temp']
                api_model_name = jsonfile['api_model_name']
                api_token_max = jsonfile['api_token_max']
                return render_template('upload.html',api_key=api_key, api_temp=api_temp, api_model_name=api_model_name, api_token_max=api_token_max,files=files)


if __name__ == '__main__':
    app.run(debug=True)



# # construct_index(api_key=="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",api_temp=0.7,api_model_name="gpt-3.5-turbo",api_token_max=2000)
# # print("embedding done")

# try:
#     print(ask_ai("Wha is codecanyon?"))
# except Exception as e:
#     print("error----->")
#     print(e)
#     print("error end----->")