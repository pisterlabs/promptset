from flask import Flask, jsonify
import os
from openai import OpenAI
import time
# Set the OpenAI API Key in the environment
os.environ["OPENAI_API_KEY"] = "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"
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
# Retrieve an existing assistant
assistant = client.beta.assistants.retrieve('asst_qcQpDO06iKVKqdbMKNuaQoQv')
# Create the Flask application
app = Flask(__name__)
# Define a route to get the assistant's reply
@app.route('/get_reply', methods=['GET'])
def get_reply():
    # Create a thread and send a message to the assistant as before, then return the last reply
    thread = client.beta.threads.create()
    client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content="What should I do at workplace to improve mental health"
    )
    
    # Creating an Run
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id
    )
    # Poll the run until it is completed
    def poll_run(run, thread):
        while run.status != "completed":
            run = client.beta.threads.runs.retrieve(
                thread_id=thread.id,
                run_id=run.id,
            )
            time.sleep(0.5)
        return run
    run = poll_run(run, thread)
    # Extracting the message
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    # Check if there's an assistant reply and return it
    for m in messages:
        if m.role == "assistant":
            return jsonify({"reply": m.content[0].text.value})
    return jsonify({"reply": "No reply from assistant."})
if __name__ == '__main__':
    app.run(debug=True)