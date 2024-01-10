from flask import Flask, jsonify, request, abort
import os
from router import handle_slash_command
from my_classes import SlackRequestData
from slack_verification import OpenAIChatHandler
import logging
import sys
import json



app = Flask(__name__)
secret = os.environ.get('slack_signing_secret')

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

@app.route('/', methods=['POST'])
def handle_request():
    logging.info(f'Handling request. my sig: {secret}')
    inbound_data = SlackRequestData(request.form, request.headers)
    # The body of the request needs to be obtained from request.get_data() or similar
    body = request.get_data().decode('utf-8')
    logging.info('Class inbound data')
    logging.info(inbound_data.to_dict())
    logging.info('Raw body')
    logging.info(body)
    #print(inbound_data.to_dict())


    logging.info('Sending payload to Verifying signature')
    logging.info('x-slack-signature: ' + inbound_data.signature)
    # Pass the secret, the inbound_data object, and the raw body to the verification function
    if not verify_slack_signature(secret, inbound_data, body):
        logging.info('Signature verification failed')
        abort(403)  # Forbidden if the signature verification fails

    # Handle the slash command
    return jsonify({"status": "success"})

@app.route('/')
def index():
    return jsonify({"Choo Choo": "Welcome to your Flask app 🚅"})


if __name__ == '__main__':
    app.logger.addHandler(logging.StreamHandler(sys.stdout))
    app.logger.setLevel(logging.ERROR)
    app.run(debug=True, port=os.getenv("PORT", default=5000))
