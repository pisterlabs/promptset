from flask import Flask, request, jsonify
from OpenAIAuth import Authenticator

app = Flask(__name__)

email = "your_email"
password = "your_password"
proxy = None

auth = Authenticator(email_address=email, password=password, proxy=proxy)

@app.route('/auth', methods=['POST'])
def get_token():
    req_data = request.get_json()
    email = req_data['email']
    password = req_data['password']
    
    auth.email_address = email
    auth.password = password
    
    try:
        auth.begin()
    except:
        return jsonify({"error": "Authentication failed"}), 401
    
    return jsonify({"access_token": auth.access_token}), 200

if __name__ == '__main__':
    app.run(debug=True)
