import openai
import socket
import threading
import json
import os

from dotenv import load_dotenv


load_dotenv()

# Set your OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

def process_story(story):

    model = "gpt-4-1106-preview"
    messages = [
        {"role": "system", "content": "you are a lie detector about storytelling\n\nyou need to decide if the story is true or a lie"},
        {"role": "user", "content": story}
    ]
    functions = [
        {
            "name": "true_story",
            "description": "Check if the story is true",
            "parameters": {
                "type": "object",
                "properties": {
                    "isTrue": {"type": "boolean"},
                    "accuracy": {"type": "number"},
                    "reason": {"type": "string"}
                },
                "required": ["isTrue", "accuracy", "reason"]
            },
        }
    ]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        functions=functions,
        function_call={
            "name": "true_story",
        },
        max_tokens=150
    )

    # Extract relevant information from the OpenAI response
    output = response['choices'][0]['message']["function_call"]["arguments"]
    
    return json.loads(output)

async def handle_client(stream):
    async with stream:
        while True:
            data = await stream.receive_some(1024)
            if not data:
                break

            # Assuming data is received as JSON
            try:
                request = json.loads(data.decode())
                story = request.get("story", "")
            except json.JSONDecodeError:
                response = {
                    "error": "Invalid JSON format",
                }
                await stream.send_all(json.dumps(response).encode())
                continue

            # Process the story with ChatGPT and get the JSON response
            result = process_story(story)

            # Send the result back to the client
            await stream.send_all(json.dumps(result).encode())

def test():
    # Test with a sample story
    test_story = "I once traveled to the bottom of the ocean and befriended a giant squid."
    test_story = "I once drove a car and it was fun."

    # Process the story with ChatGPT
    result = process_story(test_story)

    # Print the result
    print("Test Result:")
    print(result)
    
def handle_client(client_socket):
    # Receive data from the client
    data = client_socket.recv(1024).decode()
    request = json.loads(data)

    try:
        story = request.get("story", "")
    except json.JSONDecodeError:
        response = {
            "error": "Invalid JSON format",
        }
        client_socket.send(json.dumps(response).encode())
        client_socket.close()
        return

    # Process the story with ChatGPT and get the JSON response
    result = process_story(story)

    # Send the result back to the client
    client_socket.send(json.dumps(result["reason"]).encode())

    # Close the client socket
    client_socket.close()

def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('127.0.0.1', 12345))
    server.listen(5)

    print("Server listening on port 12345")

    try:
        while True:
            client_socket, addr = server.accept()
            print(f"Accepted connection from {addr}")

            # Handle each client in a separate thread
            client_handler = threading.Thread(target=handle_client, args=(client_socket,))
            client_handler.start()
    except KeyboardInterrupt:
        print("Server shutting down...")
    finally:
        server.close()
        print("Server closed.")

if __name__ == "__main__":
    start_server()