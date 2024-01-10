from flask import Flask, request, jsonify, render_template
import json
app = Flask(__name__)
import openai
openai.api_key = "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"

# Initialize a counter to keep track of the current question
current_question_index = 0

with open("questions.txt",'r') as file:
    questions = file.readlines()

questions = [json.loads(qus) for qus in questions]
answers = {}

messages = [{"role":"system","content":"You are an expert resume/profile maker, you will give with multiple questions asked to the user and corresponding answers, you need to standardize the answer as per the request to make a proper industry standard resume/CV. Consider adding related details as per the context to make the resume more informative. for all questions always provide answers in json/dictionary format with no other detail/information."},]

def get_response(qus,ans):
    global messages
    messages.append({"role":"assistant","content":qus['qus']})
    messages.append({"role":"user","content":qus['prompt']+f"\n```{ans}```"})
    response = openai.ChatCompletion.create(messages=messages,
                                            model="gpt-3.5-turbo",
                                            temperature=1.5)
    return response['choices'][0].message.content


student_profile = []
@app.route('/')
def index():
    return render_template('resume.html')

@app.route('/reset_chat', methods=['POST'])
def reset_chat():
    global current_question_index, answers, student_profile, messages
    current_question_index = 0
    answers = {}
    student_profile = []
    messages = [{"role":"system","content":"You are an expert resume/profile maker, you will give with multiple questions asked to the user and corresponding answers, you need to standardize the answer as per the request to make a proper industry standard resume/CV. Consider adding related details as per the context to make the resume more informative. for all questions always provide answers in json/dictionary format with no other detail/information. \n make sure to note that you can not ask any questions back to the user only extract information from text provided, if info not present use Null as the vlue for the respective key \n also correct spellings if wrong in the user response."},]
    return jsonify({'message': 'Chat reset successful'})

# ... (rest of the code remains unchanged)

@app.route('/send_message', methods=['POST'])
def send_message():
    global current_question_index, answers
    # Retrieve the user's answer from the frontend
    user_message = request.json['message']
    print(user_message)

    

    # If there are questions left to ask
    if current_question_index > 0:
        previous_question = questions[current_question_index - 1]
        answers[previous_question['qus']] = user_message
        details = get_response(previous_question,user_message)
        student_profile.append(details)
        print(details)
        ## call the function to export data into a word file and then convert word to pdf
        #
        ##
        #

    # Check if there are more questions to ask
    if current_question_index < len(questions):
        next_question = questions[current_question_index]["qus"]
        current_question_index += 1
    else:
        next_question = "No more questions. Thank you!"
        print(student_profile)
        # Optionally, write the answers to a text file

    return jsonify({'question': next_question})

if __name__ == '__main__':
    app.run(debug=True)
