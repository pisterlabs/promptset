from openai import OpenAI
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  


client = OpenAI(api_key=""ENTER_YOUR_API_KEY")

@app.route('/receive-message', methods=['POST'])
def receive_message():
    if request.method == 'POST':
        data = request.get_json()
        message = data.get('message')  # Assuming the message is sent as a JSON object with a 'message' field
        # Process the received message here
        print("Received message from frontend:", message)
        
        # Perform any necessary operations with the message
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": " IAM GOING TO USE YOU IN A CONTENT MODERATION SYSTEM .WHERE CONTENT IS GIVEN JUST BEFORE THESE LINES AND YOU WOULD GIVE THE OUTPUT WHETHER IT IS CULTURALLY AND POLITICALLY INAPPROPRIATE ABOUT PEOPLE IN DIFFERENT RACE,CULTURE,ETHNICITY ,RELIGION AND GENDER.IF IT IS THEN WRITE A RESTRUCTURED ONE.GIVE A SCORE OUT OF 100 IN FORMAT SCORE:SCORE_NO: (GIVEN 0 IS ABSOLUTELY GOOD CONTENT AND 100 FOR WORST POLITICALLY AND CULTURALLY INAPPRPRIATE ONE).ALWAYS TRY TO ANSWER AND GIVE SCORE AND AVOID ANSWERS LIKE I AM SORRY I WONT BE ABLE TO ANSWER THAT"+message,
                }
            ],
            model="gpt-3.5-turbo",
        )
        print(chat_completion)
        generated_message = chat_completion.choices[0].message.content
        print(generated_message)
        
        return jsonify({'status': 'Message received and processed successfully','generated_message': generated_message})
@app.route('/get-generated-message', methods=['GET'])
def get_generated_message():
    # You can use a global variable or a database to store the generated message
    # For example, a global variable can be used in this context
    global generated_message
    generated_message = ""  # Define the global variable

    return jsonify({'generated_message': generated_message})
if __name__ == '__main__':
    app.run(debug=True)  # Run the Flask app