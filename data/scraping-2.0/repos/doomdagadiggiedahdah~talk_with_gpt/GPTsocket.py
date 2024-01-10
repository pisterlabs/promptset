#!/bin/python
import socket
import openai

# OpenAI API key
openai.api_key_path = "/home/mat/Documents/ProgramExperiments/openAIapiKey"

# Setup socket
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", 1234))
s.listen(1)
conn, addr = s.accept()

try:
    welcome = "Welcome to GPT-3. Please enter your prompt:\n"
    conn.send(welcome.encode())

    while True:
        # 1. Get the prompt from the user
        prompt = conn.recv(1024)
        if prompt.lower() == "bye":
            break
        print(prompt)
        prompt = prompt.decode()

        # 2. Send the prompt to GPT
        send = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo-1106",
            messages=[
                {"role": "system", "content": "Hello, can you help me with the following problem?"},
                {"role": "user", "content": prompt}
        ])

        response = str("\n" + send["choices"][0]["message"]["content"] + "\n\n")

        # 3. Get the response from GPT
        print(response)

        # 4. Send the response to the user
        conn.send(response.encode())

    # Close connection
    conn.close()
finally:
    # Close socket
    s.close()
