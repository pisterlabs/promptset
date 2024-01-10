from flask import Blueprint, jsonify
import openai

interaction_blueprint = Blueprint("interaction", __name__)

def generate_response(prompt):
    response = openai.Completion.create(
        engine="text_davinci-002",
        prompt=prompt,
        max_tokens=150,
        n = 1,
        stop = None,
        temperature = 0.7,
    )
    return response.choices[0].text.strip()

@interaction_blueprint.route('/interact', methods=['POST'])
def interact_endpoint(user_input, conversation_history):
    if not user_input:
        return jsonify({"error": "Please provide user input in the request body."}), 400

    # Concatenate the conversation history into a single string
    formatted_conversation = ""
    for message in conversation_history:
        formatted_conversation += f"{message['role']}: {message['content']}\n"

    # Create a prompt using the conversation history and user input
    prompt = f"{formatted_conversation}User: {user_input}\nAssistant:"
    response = generate_response(prompt)

    # Append the user input and assistant response to the conversation history
    conversation_history.append({"role": "user", "content": user_input})
    conversation_history.append({"role": "assistant", "content": response})

    return jsonify({"response": response, "conversation_history": conversation_history})