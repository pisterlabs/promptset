from flask import Flask, render_template_string, request
import openai
import os
import pickle

app = Flask(__name__)

# Set openAI key:
openai.api_key = os.getenv("OPENAI_API_KEY")


# Read HTML template:
with open('index.html', 'r') as file:
    html_template = file.read()


# Read examInfo:
# Open the file for reading
with open('exam_info.pkl', 'rb') as f:
    # Load the dictionary from the file
    examInfo_dict = pickle.load(f)


def generate_randomTopic(exam):
    # Initialize the conversation with the assistant
    conversation = [
        {"role": "system", "content": "You are a helpful assistant that generates a random writing topic for english exams."},
        {"role": "user", "content": f"Give one writing topic for this english exam: {exam}."},
        {"role": "user", "content": "Do not add anything rather than the topic."}
    ]

    # Send the conversation to the API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation
    )

    # Extract the cover letter from the response
    writingtopic = response['choices'][0]['message']['content']

    return writingtopic


def evaluateWriting(exam, topic, writing):
    # Initialize the conversation with the assistant
    conversation = [
        {"role": "system", "content": "You are a helpful assistant that gives a comprehensive feedback to the writing task for the given english exam."},
        {"role": "user", "content": "Evaluate the following writing sample for its grammar, structure, and coherence. The writing sample is from an English learner who wants to pass the writing section of an English exam."},
        {"role": "user", "content": "give an estimation of the score based on the exam requirements."},
        {"role": "user", "content": f"Exam: {exam}"},
        {"role": "user", "content": f"topic: {topic}"},
        {"role": "user", "content": f"Writing: {writing}"}
    ]
        

    # Send the conversation to the API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=conversation
    )

    # Extract the cover letter from the response
    feedback = response['choices'][0]['message']['content']

    return feedback


@app.route('/')
def index():
    exam_names = list(examInfo_dict.keys())
    
    return render_template_string(
    html_template,
    exams=exam_names,
    examInfo_dict=examInfo_dict)

@app.route('/random_topic')
def random_topic():
  # Get the selected exam from the request parameters
  exam = request.args.get('exam')

  # Generate a random topic using OpenAI APIs
  writingtopic = generate_randomTopic(exam)

  # Return the topic as a response
  return writingtopic

@app.route('/submitWriting', methods=['POST'])
def submitWriting():
    # Retrieve values from the form
    exam = request.form['exam-select']
    topic = request.form['topic']
    writing = request.form['writing']
    
    # Process the writing and generate feedback
    feedback = evaluateWriting(exam=exam, topic=topic, writing=writing)
    # Return the feedback as a JSON response
    return feedback
    
if __name__ == '__main__':
    app.run(debug=True)
