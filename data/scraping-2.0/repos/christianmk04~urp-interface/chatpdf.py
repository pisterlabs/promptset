# IMPORT FLASK APP DEPENDENCIES
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from os import environ
from distutils.log import debug
from fileinput import filename

app = Flask(__name__)
CORS(app)

##################################################################################################################################################################################################
##################################################################################################################################################################################################
##################################################################################################################################################################################################
'''
FUNCTIONS HERE ARE TO BE USED FOR RENDERING OF THE DIFFERENT HTML PAGES OF THE INTERFACE
'''

# RENDER KEYWORD AND TEXT INPUT PAGE
@app.route('/')
def main_text():
    return render_template('interface.html')

# RENDER UPLOAD PDF PAGE
@app.route('/upload')
def main_upload():
    return render_template('interface_upload.html')

# RENDER RETRIEVAL PAGE FOR SELECTION
@app.route('/retrieval')
def main_retrieval_page():
    return render_template('interface_retrieval.html')

# RENDER PAGE TO EDIT CS AND RELATED QA
@app.route('/retrieval_csqa')
def main_retrieval_csqa_page():
    return render_template('retrieval_csqa.html')

##################################################################################################################################################################################################
##################################################################################################################################################################################################
##################################################################################################################################################################################################

# IMPORT LANGCHAIN DEPENDENCIES
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
import tiktoken

from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

import openai
import requests
import os 

##################################################################################################################################################################################################
##################################################################################################################################################################################################
##################################################################################################################################################################################################
'''
FUNCTIONS HERE ARE TO BE USED FOR GENERATION OF RESOURCES USING PDF FILES

FUNCTIONS IN THIS SECTION INCLUDE:

REUSABLE FUNCTION - PDF_READ 
- USED TO READ THE PDF FILE AND GENERATE THE KNOWLEDGE BASE AND CHAIN FOR GENERATION OF CASE STUDY

REUSABLE FUNCTION - UPLOAD_FILE_SKELETON
- USED TO UPLOAD THE PDF FILE TO THE DB

REUSABLE FUNCTION - UPLOAD_CS_SKELETON
- USED TO UPLOAD THE GENERATED CASE STUDY TO THE DB

FUNCTION - UPLOAD_CS
- GENERATES CASE STUDY AND UPLOADS IT TO THE DB

FUNCTION - UPLOAD_QA
- GENERATES INDEPENDENT QA AND UPLOADS IT TO THE DB

FUNCTION - UPLOAD_CSQA
- GENERATES CASE STUDY AND RELATED QA AND UPLOADS IT TO THE DB
'''

# FUNCTION TO DO THE PDF READING - REUSABLE FUNCTION
def pdf_read(uploaded_file):
    reader = PdfReader(uploaded_file)

    raw_text = ""
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text

    text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap = 200,
        length_function = len,
    )
    texts = text_splitter.split_text(raw_text)

    embeddings = OpenAIEmbeddings()

    knowledge_base = FAISS.from_texts(texts, embeddings)

    chain = load_qa_chain(ChatOpenAI(), chain_type="stuff")

    return [chain, knowledge_base]

# FUNCTION TO UPLOAD PDF FILE TO DB - REUSABLE FUNCTION
def upload_file_skeleton(file_to_upload, file_name):
    mongo_upload_endpoint = "http://localhost:5001/upload_pdf" + "/" + file_to_upload.filename
    try: 
        response = requests.post(mongo_upload_endpoint, file_to_upload)
        file_id = response.text
        print("Successfully uploaded file to DB")
    except Exception as e:
        print("Error")
        print(e)
    
    return file_id

# FUNCTION TO UPLOAD GENERATED CASE STUDY TO DB
def upload_cs_skeleton(file_id, case_study_output, topics):
    mongo_upload_endpoint = "http://localhost:5001/upload_cs_for_pdf" + "/" + file_id

    topic_split = topics.split("\n")
    main_topic = topic_split[0].split(":")[1].strip()
    sub_topics = topic_split[1].split(":")[1].strip()

    case_study_output_json = {
        "main_topic": main_topic,
        "sub_topic": sub_topics,
        "case_study": case_study_output
    }

    try: 
        response = requests.post(mongo_upload_endpoint, json=case_study_output_json)
        print("Successfully uploaded case study to DB")
        cs_id = response.text
    except Exception as e:
        print("Error")
        print(e)
    
    return cs_id

# UPLOAD PDF FILE TO DB AND GENERATE + UPLOAD CASE STUDY TO DB
@app.route('/upload_file_cs', methods=['POST'])
def upload_cs():
    
    print('-----------------Uploading file------------------------')

    # ERROR HANDLING - TO MAKE SURE THAT A FILE HAS BEEN UPLOADED BY THE USER AND A VALID API KEY IS ENTERED

    user_api_key = request.form['user_api_key']
    
    if user_api_key == '':
        return render_template('interface_upload_error.html', error_message="Unable to proceed. Please enter a valid API key!")

    uploaded_file = request.files['file']
    
    if uploaded_file.filename == '':
        return render_template('interface_upload_error.html', error_message="Unable to proceed. Please upload a PDF file!")
    
    # SET API KEY FOR GENERATION OF RESOURCE

    os.environ["OPENAI_API_KEY"] = user_api_key
    
    # UPLOAD FILE TO DB
    file_id = upload_file_skeleton(uploaded_file, uploaded_file.filename)
    
    # GENERATE CASE STUDY
    chain = pdf_read(uploaded_file)[0]
    knowledge_base = pdf_read(uploaded_file)[1]

    cs_query = "Based on the contents in this file, can you create a fictional case study for me about a fictional company? The case study should revolve around Agile and DevOps, and should reference as much of the contents of in the file. The case study should follow this structure: 1. Introduction of Company and Background 2. Current Practices 2. Problems faced due to current practices 3. The need to implement new practices and what they are 4. Results 5. Conclusion. \n\n Make the case study in such a way where the individual sections are not numbered and that the whole case study flows seamlessly \n\n Skip the pleasantries of acknowledging the user and start generating the case study immediately (Meaning, do not start with 'Sure, here's a case study for...' or 'Here's a case study for...')."

    cs_docs = knowledge_base.similarity_search(cs_query)
    cs_output = chain.run(input_documents=cs_docs,question=cs_query)

    topic_query = f'Based on the contents in this file, can you identify the main topic of the file contents? The main topic should be a single word, and should be strictly either Agile or DevOps. Identify also, only 5 subtopics that are related to the main topic. The subtopics should be single words as well. \n\n Skip the pleasantries of acknowledging the user and start generating the topic immediately (Meaning, do not start with "Sure, here\'s the topic for..." or "Here\'s the topic for..."). Some examples of sub-topics include automation, continuous integration, continuous delivery, etc. Generate your response as follows in the example delimited by the double apostrophes: \n\n """ \n Main Topic: DevOps \n Sub-Topics: Automation, Continuous Integration, Continuous Delivery, Continuous Deployment, Continuous Testing"""'

    topic_docs = knowledge_base.similarity_search(topic_query)
    topic_output = chain.run(input_documents=topic_docs,question=topic_query)

    # UPLOAD CASE STUDY TO DB
    upload_cs_skeleton(file_id, cs_output, topic_output)

    return render_template('interface_post_upload_cs.html', cs_output=cs_output)

# UPLOAD PDF FILE TO DB AND GENERATE + UPLOAD QUESTIONS & ANSWERS TO DB
@app.route('/upload_file_qa', methods=['POST'])
def upload_qa():
    
    print('-----------------Uploading file------------------------')

    user_api_key = request.form['user_api_key']
    
    if user_api_key == '':
        return render_template('interface_upload_error.html', error_message="Unable to proceed. Please enter a valid API key!")

    uploaded_file = request.files['file']
    
    if uploaded_file.filename == '':
        return render_template('interface_upload_error.html', error_message="Unable to proceed. Please upload a PDF file!")
    
    # SET API KEY FOR GENERATION OF RESOURCE

    os.environ["OPENAI_API_KEY"] = user_api_key
    
    # UPLOAD FILE TO DB
    file_id = upload_file_skeleton(uploaded_file, uploaded_file.filename)

    # GENERATE QUESTIONS AND ANSWERS
    chain = pdf_read(uploaded_file)[0]
    knowledge_base = pdf_read(uploaded_file)[1]

    ques_ind_query = "Based on the contents of the file, can you write me 10 questions that relate to DevOps and Agile? Have the questions reference as much of the content inside the file as possible, whilst adhering to the theme of DevOps and Agile. \n\n Write the questions in the following format: \n1. Question 1\n2. Question 2\n3. Question 3 \n\n and so on. \n\n Skip the pleasantries of acknowledging the user and start generating the questions immediately (Meaning, do not start with 'Sure, here's a questions for...')."
    q_docs = knowledge_base.similarity_search(ques_ind_query)
    q_output = chain.run(input_documents=q_docs,question=ques_ind_query)

    ans_ind_query = f'Please provide the answers to the following questions. \n\n {q_output} \n\n Skip the pleasantries of acknowledging the user and start generating the answers immediately (Meaning, do not start with "Sure, here\'s the answers for...").'
    a_docs = knowledge_base.similarity_search(ans_ind_query)
    a_output = chain.run(input_documents=a_docs,question=ans_ind_query)

    topic_query = f'Based on the contents of the file, can you identify the main topic of the questions and answers? The main topic should be a single word, and should be strictly either Agile or DevOps. Identify also, only 5 subtopics that are related to the main topic. The subtopics should be single words as well. \n\n Skip the pleasantries of acknowledging the user and start generating the topic immediately (Meaning, do not start with "Sure, here\'s the topic for..." or "Here\'s the topic for..."). Some examples of sub-topics include automation, continuous integration, continuous delivery, etc. Generate your response as follows in the example delimited by the double apostrophes: \n\n """ \n Main Topic: DevOps \n Sub-Topics: Automation, Continuous Integration, Continuous Delivery, Continuous Deployment, Continuous Testing"""'

    topic_docs = knowledge_base.similarity_search(topic_query)
    topic_output = chain.run(input_documents=topic_docs,question=topic_query)

    topic_split = topic_output.split("\n")
    main_topic = topic_split[0].split(":")[1].strip()
    sub_topics = topic_split[1].split(":")[1].strip()


    # UPLOAD QUESTIONS AND ANSWERS TO DB
    mongo_upload_endpoint = "http://localhost:5001/upload_qa_for_pdf" + "/" + file_id
    qa_json = {
        "questions": q_output,
        "answers": a_output,
        "main_topic": main_topic,
        "sub_topic": sub_topics
    }
    try:
        response = requests.post(mongo_upload_endpoint, json=qa_json)
        print(response.text)
    except Exception as e:
        print("Error")
        print(e)

    return render_template('interface_post_upload_qa.html', q_output=q_output, a_output=a_output)
        
# UPLOAD PDF FILE TO DB AND GENERATE + UPLOAD CASE STUDY AND QUESTIONS & ANSWERS TO DB
@app.route('/upload_file_csqa', methods=['POST'])
def upload_csqa():
    
    print('-----------------Uploading file------------------------')

    user_api_key = request.form['user_api_key']
    
    if user_api_key == '':
        return render_template('interface_upload_error.html', error_message="Unable to proceed. Please enter a valid API key!")

    uploaded_file = request.files['file']
    
    if uploaded_file.filename == '':
        return render_template('interface_upload_error.html', error_message="Unable to proceed. Please upload a PDF file!")
    
    # SET API KEY FOR GENERATION OF RESOURCE

    os.environ["OPENAI_API_KEY"] = user_api_key
    
    # UPLOAD FILE TO DB
    file_id = upload_file_skeleton(uploaded_file, uploaded_file.filename)

    # GENERATE CASE STUDY
    chain = pdf_read(uploaded_file)[0]
    knowledge_base = pdf_read(uploaded_file)[1]

    cs_query = "Based on the contents in this file, can you create a fictional case study for me about a fictional company? The case study should revolve around Agile and DevOps, and should reference as much of the contents of in the file. The case study should follow this structure: 1. Introduction of Company and Background 2. Current Practices 2. Problems faced due to current practices 3. The need to implement new practices and what they are 4. Results 5. Conclusion. \n\n Make the case study in such a way where the individual sections are not numbered and that the whole case study flows seamlessly \n\n Skip the pleasantries of acknowledging the user and start generating the case study immediately (Meaning, do not start with 'Sure, here's a case study for...' or 'Here's a case study for...')."

    cs_docs = knowledge_base.similarity_search(cs_query)
    cs_output = chain.run(input_documents=cs_docs,question=cs_query)

    topic_query = f'Based on the contents of the file, can you identify the main topic of the questions and answers? The main topic should be a single word, and should be strictly either Agile or DevOps. Identify also, only 5 subtopics that are related to the main topic. The subtopics should be single words as well. \n\n Skip the pleasantries of acknowledging the user and start generating the topic immediately (Meaning, do not start with "Sure, here\'s the topic for..." or "Here\'s the topic for..."). Some examples of sub-topics include automation, continuous integration, continuous delivery, etc. Generate your response as follows in the example delimited by the double apostrophes: \n\n """ \n Main Topic: DevOps \n Sub-Topics: Automation, Continuous Integration, Continuous Delivery, Continuous Deployment, Continuous Testing"""'

    topic_docs = knowledge_base.similarity_search(topic_query)
    topic_output = chain.run(input_documents=topic_docs,question=topic_query)

    # UPLOAD CASE STUDY TO DB
    cs_id = upload_cs_skeleton(file_id, cs_output, topic_output)

    # GENERATE QUESTIONS AND ANSWERS
    ques_cs_query = f'Based on the case study below, can you create 10 questions about the case study? Phrase them in a way where it will require more critical thinking. \n\n Case Study: {cs_output} \n\n Skip the pleasantries of acknowledging the user and start generating the questions immediately (Meaning, do not start with \'Sure, here\'s a questions for...\')'
    q_docs = knowledge_base.similarity_search(ques_cs_query)
    q_output = chain.run(input_documents=q_docs,question=ques_cs_query)

    a_query = f'Based on the case study and the questions below, could you provide the answers to the questions? \n\n Case Study: {cs_output} \n\n Questions: {q_output} \n\n Skip the pleasantries of acknowledging the user and start generating the answers immediately. (Meaning, do not start with "Sure, here\'s the answers for...").'
    a_docs = knowledge_base.similarity_search(a_query)
    a_output = chain.run(input_documents=a_docs,question=a_query)

    topic_split = topic_output.split("\n")
    main_topic = topic_split[0].split(":")[1].strip()
    sub_topics = topic_split[1].split(":")[1].strip()

    # UPLOAD RELATED QUESTIONS AND ANSWERS TO DB
    mongo_upload_endpoint = "http://localhost:5001/upload_csqa_for_pdf" + "/" + cs_id
    qa_json = {
        "questions": q_output,
        "answers": a_output,
        "main_topic": main_topic,
        "sub_topic": sub_topics
    }
    try:
        response = requests.post(mongo_upload_endpoint, json=qa_json)
        print(response.text)
    except Exception as e:
        print("Error")
        print(e)

    return render_template('interface_post_upload_csqa.html', cs_output=cs_output, q_output=q_output, a_output=a_output)

##################################################################################################################################################################################################
##################################################################################################################################################################################################
##################################################################################################################################################################################################
'''
FUNCTIONS HERE ARE TO BE USED FOR API CALLS FROM OTHER SOURCES SUCH AS POSTMAN OR OTHER INTERFACES

FUNCTIONS IN THIS SECTION INCLUDE:

- CASE STUDY
    - GENERATE CASE STUDY (api_get_cs)

- INDEPENDENT QUESTIONS AND ANSWERS
    - GENERATE QUESTIONS AND ANSWERS (api_get_qa)

- CASE STUDY QUESTIONS AND ANSWERS
    - GENERATE CASE STUDY + RELATED QUESTIONS AND ANSWERS (api_get_csqa)
'''

# API ROUTES FOR OTHER APPLICATIONS TO CALL AND USE 

# API ROUTE TO GENERATE CASE STUDY
@app.route('/api_get_cs/<string:api_key>/<string:main_topic>/<string:sub_topic>', methods=['GET'])
def api_get_cs(api_key, main_topic, sub_topic):

    # SET UP MONGO RETRIEVAL FROM MONGO MICROSERVICE
    mongo_retrieve_endpoint = "http://localhost:5001/get_case_study/manual/" + main_topic + "/" + sub_topic
    try:
        response = requests.get(mongo_retrieve_endpoint)
    except Exception as e:
        print("Error")
        print(e)

        # GET DATA FROM MONGO MICROSERVICE RESPONSE
    json_data = response.json()
    data = json_data["data"][0]

    ref_case_study = data["content"]

    # SET API KEY - CHECK IF API KEY IS VALID OR ENTERED
    openai.api_key = api_key

    if api_key == '':
        return jsonify({"error": "Unable to proceed. Please enter in API key!"})
    
    # GENERATE CHAT COMPLETION
    try:
        completion = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo",
            messages = [
                {"role": "system", "content": "You are an instructor teaching an Agile and DevOps course, your job is to provide questions and answers for students for the purpose of assessing students purposes. You are currently chatting with a Professor of the course, who is asking you for questions and answers about Agile and DevOps."},
                {"role": "user", "content": f"Can you provide me with a sample case study about {main_topic} that focuses on {sub_topic}? Skip the pleasantries of acknowledging the user and start generating the case study immediately. (Meaning, do not start with 'Sure, here's a case study for...')."},
                {"role": "assistant", "content": f"{ref_case_study}"},
                {"role": "user", "content": f"Please provide me with another case study about {main_topic} that focuses on {sub_topic} following the same format as what you have just generated. Skip the pleasantries of acknowledging the user and start generating the case study immediately as before. (Meaning, do not start with 'Sure, here's a case study for...')"},
            ],
            temperature = 1.1,
            max_tokens = 2048,
        )
    except Exception as e:
        return jsonify({"error": e})
    
    generated_case_study = completion.choices[0].message.content

    # UPLOAD CASE STUDY TO DB
    mongo_upload_endpoint = "http://localhost:5001/upload_cs"
    cs_json = {
        "content": generated_case_study,
        "main_topic": main_topic,
        "sub_topic": sub_topic,
        "mode": "api_call"

    }

    try:
        response = requests.post(mongo_upload_endpoint, json=cs_json)
        print(response.text)
    except Exception as e:
        print("Error")
        print(e)
    
    return jsonify(
        {
            "case study": generated_case_study,
            "message" : f"Case study generated for {main_topic} focusing on {sub_topic}. Case study uploaded to database."
        }
    )



# API TO GENERATE QUESTIONS AND ANSWERS
@app.route('/api_get_qa/<string:api_key>/<string:sub_topic>', methods=['GET'])
def api_get_qa(api_key, sub_topic):

    sub_topics = ["Automation", "Software Design", "Version Control", "Software Lifecycle", "Agile Methodologies", "Software Security"]

    # SET UP MONGO RETRIEVAL FROM MONGO MICROSERVICE
    mongo_retrieve_endpoint = "http://localhost:5001/get_ind_questions" + "/manual" + "/" + sub_topic
    try:
        response = requests.get(mongo_retrieve_endpoint)
    except Exception as e:
        print("Error")
        print(e)

    # FORMAT QUESTIONS AND ANSWERS INTO STRING TO BE PUT INTO THE CHAT COMPLETION MESSAGE 
    questions_string = ""
    answers_string = ""

        # GET DATA FROM MONGO MICROSERVICE RESPONSE
        # DATA RETRIEVED IS THE REFERENCE QUESTIONS AND ANSWERS
    json_data = response.json()
    data = json_data["data"]

        # FORMAT QUESTIONS AND ANSWERS 
    for i in range(len(data)):
        questions_string += f'{i+1}. ' + data[i]["question"] + "\n"
        answers_string += f'{i+1}. ' + data[i]["answer"] + "\n"

    print(questions_string)
    print(answers_string)

    # SET API KEY - CHECK IF API KEY IS VALID OR ENTERED
    openai.api_key = api_key

    if api_key == '':
        return jsonify({"error": "Unable to proceed. Please enter in API key!"})
    
    # GENERATE CHAT COMPLETION
    try:
        completion = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo",
            messages = [
                {"role": "system", "content": "You are an instructor teaching an Agile and DevOps course, your job is to provide questions and answers for students for the purpose of assessing students purposes. You are currently chatting with a Professor of the course, who is asking you for questions and answers about Agile and DevOps. "},
                {"role": "user", "content": f"Can you provide me with sample questions and answers about {sub_topic} under Agile/DevOps? Provide the questions and answers in a way where it will require more critical thinking. Format your response in this way:\n\n 'Questions: \n1.\n2.\n3. \n\n Answers: \n1.\n2.\n3.' \n\n Skip the pleasantries of acknowledging the user and start generating the questions and answers immediately. (Meaning, do not start with 'Sure, here's a questions and answers for...')"},
                {"role": "assistant", "content": f"Questions:\n{questions_string}\nAnswers:\n{answers_string}"},
                {"role": "user", "content": "Please provide me with 10 more questions and answers following the same format as what you have just generated. Skip the pleasantries of acknowledging the user and start generating the questions and answers immediately. (Meaning, do not start with 'Sure, here's a questions and answers for...')"},
            ],
            temperature = 1.1,
            max_tokens = 2048,
        )
    except Exception as e:
        return jsonify({"error": e})
    
    answers_unformatted = completion.choices[0].message.content.split("Answers:")[1]
    questions_unformatted = completion.choices[0].message.content.split("Answers:")[0].split("Questions:")[1]
    
    mongo_upload_endpoint = "http://localhost:5001/upload_ind_qa"
    qa_json = {
        "mode": "api_call",
        "sub_topic": sub_topic,
        "questions": questions_unformatted,
        "answers": answers_unformatted
    }
    try:
        response = requests.post(mongo_upload_endpoint, json=qa_json)
        print(response)
    except Exception as e:
        print("Error")
        print(e)

    questions_formatted_arr = []
    answers_formatted_arr = []

    questions_split_arr = questions_unformatted.split("\n")
    answers_split_arr = answers_unformatted.split("\n")

    for i in range(len(questions_split_arr)):
        if questions_split_arr[i] != '':
            questions_formatted_arr.append(questions_split_arr[i])
    
    for i in range(len(answers_split_arr)):
        if answers_split_arr[i] != '':
            answers_formatted_arr.append(answers_split_arr[i])

    return jsonify(
        {
            "questions" : questions_formatted_arr,
            "answers" : answers_formatted_arr,
            "message" : f"Questions and answers generated for {sub_topic}. Uploaded generated questions and answers to the database."

        }
    )


# API ENDPOINT TO GENERATE CASE STUDY, QUESTIONS AND ANSWERS
@app.route('/api_get_csqa/<string:api_key>/<string:main_topic>/<string:sub_topic>', methods=['GET'])
def api_get_csqa(api_key, main_topic, sub_topic):

    # CHECK IF SUB_TOPIC IS IN THE LIST OF SUB_TOPICS
    sub_topics = ["Automation", "Software Design", "Version Control", "Software Lifecycle", "Agile Methodologies", "Software Security"]
    if sub_topic not in sub_topics:
        # SET UP MONGO RETRIEVAL FROM MONGO MICROSERVICE
        mongo_retrieve_endpoint = "http://localhost:5001/get_csqa/manual/" + main_topic + "/" + sub_topic
    else:
        # SET UP MONGO RETRIEVAL FROM MONGO MICROSERVICE
        mongo_retrieve_endpoint = "http://localhost:5001/get_csqa/automatic/" + main_topic + "/" + sub_topic
    
    try:
        response = requests.get(mongo_retrieve_endpoint)
        data = response.json()
    except Exception as e:
        print("Error")
        print(e)
    
    case_study = data["case_study"]
    questions = data["questions"]
    answers = data["answers"]

    # FORMAT QUESTIONS AND ANSWERS INTO STRINGS
    questions_string = ""
    answers_string = ""

    for i in range(len(questions)):
        questions_string += f'{i+1}. ' + questions[i] + "\n"
        answers_string += f'{i+1}. ' + answers[i] + "\n"

    # SET API KEY - CHECK IF API KEY IS VALID OR ENTERED
    openai.api_key = api_key

    if api_key == '':
        return jsonify({"error": "Unable to proceed. Please enter in API key!"})

    # GENERATE CHAT COMPLETION
    try:
        completion = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo",
            messages = [
                {"role": "system", "content": "You are an instructor teaching an Agile and DevOps course, your job is to provide questions and answers for students for the purpose of assessing students purposes. You are currently chatting with a Professor of the course, who is asking you for questions and answers about Agile and DevOps. "},
                # REFERENCE PROMPT ENGINEERING FOR CASE STUDY
                {"role": "user", "content": f"Can you provide me with a sample case study about {main_topic} that focuses on {sub_topic}? Skip the pleasantries of acknowledging the user and start generating the case study immediately. (Meaning, do not start with 'Sure, here's a case study for...')."},
                {"role": "assistant", "content": f"{case_study}"},
                # REFERENCE PROMPT ENGINEERING FOR QUESTIONS AND ANSWERS
                {"role": "user", "content": f"Can you provide me with sample questions and answers about the case study above? Where the questions are about {main_topic}, focusing on {sub_topic}? Provide the questions and answers in a way where it will require more critical thinking. Format your response in this way:\n\n 'Questions: \n1.\n2.\n3. \n\n Answers: \n1.\n2.\n3.' \n\n Skip the pleasantries of acknowledging the user and start generating the questions and answers immediately. (Meaning, do not start with 'Sure, here's a case study/questions and answers for...')"},
                {"role": "assistant", "content": f"Questions:\n{questions_string}\nAnswers:\n{answers_string}"},
                {"role": "user", "content": f"Please provide me with another case study, and 10 sample questions and sample answers for the case study above. Have the case study, questions and answers be about {main_topic} which focuses on {sub_topic}. Follow the same format as what you have just generated, such as denoted in the triple apostrophe delimiters: \n\n ''' Case Study:\n (Generated Case Study)\n\nQuestions: \n1.\n2.\n3.\n\n Answers:\n1.\n2.\n3.\n\n ''' \n\n Skip the pleasantries of acknowledging the user and start generating the questions and answers immediately. (Meaning, do not start with 'Sure, here's a case study/questions and answers for...')"},
            ],
            temperature = 1.1,
            max_tokens = 2048,
        )
    except Exception as e:
        return jsonify({"error": e})
    
    # FORMAT CASE STUDY, QUESTIONS AND ANSWERS INTO STRINGS
    content = completion.choices[0].message.content

    # GET QUESTIONS AND ANSWERS FIRST
    questions_unformatted = content.split("Answers:")[0].split("Questions:")[1]
    answers_unformatted = content.split("Answers:")[1]
    
    questions_formatted_arr = []
    answers_formatted_arr = []

    questions_split_arr = questions_unformatted.split("\n")
    answers_split_arr = answers_unformatted.split("\n")

    for i in range(len(questions_split_arr)):
        if questions_split_arr[i] != '' and questions_split_arr[i] != ' ':
            questions_formatted_arr.append(questions_split_arr[i])

    for i in range(len(answers_split_arr)):
        if answers_split_arr[i] != '' and answers_split_arr[i] != ' ':
            answers_formatted_arr.append(answers_split_arr[i])

    # GET CASE STUDY
    generated_case_study = content.split("Answers:")[0].split("Questions:")[0].split("Case Study:")[1]

    # SET UP MONGO UPLOAD CS TO MONGO MICROSERVICE
    mongo_upload_cs_endpoint = "http://localhost:5001/upload_cs"
    new_cs = {
        "main_topic" : main_topic,
        "sub_topic" : sub_topic,
        "content" : generated_case_study,
        "mode": "api_call"
    }
    try:
        response = requests.post(mongo_upload_cs_endpoint, json=new_cs)
        print(response)
    except Exception as e:
        print("Error")
        print(e)

    # SET UP MONGO UPLOAD RELATED QA TO MONGO MICROSERVICE
    mongo_upload_qa_endpoint = "http://localhost:5001/upload_qa_for_cs"
    new_qa_data = {
        "main_topic" : main_topic,
        "sub_topic" : sub_topic,
        "mode": "api_call",
        "content": generated_case_study,
        "questions": questions_unformatted,
        "answers": answers_unformatted,
    }
    try:
        response = requests.post(mongo_upload_qa_endpoint, json=new_qa_data)
        print(response)
    except Exception as e:
        print("Error")
        print(e)

    return jsonify(
        {
            "case_study" : generated_case_study,
            "questions" : questions_formatted_arr,
            "answers" : answers_formatted_arr,
            "message" : f"Case study, questions and answers generated for {main_topic} focusing on {sub_topic}. Uploaded all to the database.",
        }
    )



# FLASK APP ROUTE
if __name__ == '__main__':
    app.run(port=5000, debug=True)