import openai, os
#from app import socketio

def process_code(file_path):
    # Read the contents of the file
    with open(file_path, 'r') as code_file:
        code_contents = code_file.read()

    # Emit progress update
    #socketio.emit('progress', {'percentage': 10})

    # Set up the OpenAI API
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Emit progress update
    #socketio.emit('progress', {'percentage': 50})

    # Send a request to the OpenAI API
    message = "Improve the following code style and add comments:\n\n%s\n" % code_contents
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = [
          {"role":"user", "content":message},
        ]
    )

    # Emit progress update
    #socketio.emit('progress', {'percentage': 100})

    # Get the improved code and comments from the API response
    #improved_code = response.choices[0].text.strip()
    improved_code = response.choices[0].message.content.strip()

    # Replace the following example lines with your actual logic
    prettified_code = improved_code
    comments = "Example comments"

    # Save the prettified code to a new file
    prettified_file_path = os.path.join("uploads", "prettified_" + os.path.basename(file_path))
    with open(prettified_file_path, 'w') as prettified_code_file:
        prettified_code_file.write(prettified_code)

    return prettified_code, comments, prettified_file_path
