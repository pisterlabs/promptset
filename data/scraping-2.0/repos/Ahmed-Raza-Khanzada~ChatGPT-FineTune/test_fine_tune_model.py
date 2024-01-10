from flask import Flask, request, jsonify
import openai
from bs4 import BeautifulSoup
from api_credentials import api_key,model_id

app = Flask(__name__)
openai.api_key = api_key  

def clean_html(message):
    soup = BeautifulSoup(message, 'html.parser')
    text = soup.get_text()
    return text

def chat_with_fine_tuned_model(messages):
    response = openai.ChatCompletion.create(
        model=model_id,  
        messages=messages
    )
    reply = response['choices'][0]['message']['content']
    return reply
def check_null(var):
    if var == None:
        return "NA"
    elif type(var)==int or type(var)==float:
        return str(var)
    return var
@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Extract message from POST request
        data = request.json
        message = clean_html(data["message"])
        model = data["model"]
        manufacturer = data["manufacturer"]
        year_of_manufactured = data["year_of_manufactured"]
        ecu_number = data["ecu_number"]
        model = check_null(model)
        manufacturer= check_null(manufacturer)
        ecu_number = check_null(ecu_number)
        year_of_manufactured = check_null(year_of_manufactured)
        message = " ".join(message.split("\u00a0"))

        # Prepare messages for ChatGPT
        messages = [
            {"role": "system", "content": f"You are an assistant from staff that responds to Clients queries\nhere are their vehicle details:\nModel:{model}\nManufacturer:{manufacturer}\nYear of Manufature:{year_of_manufactured}\nECU Number:{ecu_number}\n"},
            {"role": "user", "content": message}
        ]

        # Get response from ChatGPT
        response = chat_with_fine_tuned_model(messages)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host = "0.0.0.0",port =5000 , debug=True)
