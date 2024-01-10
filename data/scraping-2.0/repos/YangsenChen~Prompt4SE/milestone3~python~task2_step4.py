import os
import glob
import openai

# Define the root directory
root_dir = "./generated_code/"
# Define the chat history directory
chat_dir = "./chat_history/task2"
os.makedirs(chat_dir, exist_ok=True)  # create chat history directory if it doesn't exist

# Your API key here
openai.api_key = "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"

# Function to send a message to chatgpt
def send_message_to_chatgpt(message, conversation_history):
    conversation_history.append({'role': 'user', 'content': message})

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {'role': 'system', 'content': 'You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture.'}
        ] + conversation_history,
    )

    assistant_message = response.choices[0].message.content
    conversation_history.append({'role': 'assistant', 'content': assistant_message})
    return assistant_message, conversation_history

# Iterate over all the folders from code_00 to code_49
for i in range(34,50):
    folder_name = f"code_{i:02d}"
    target_folder = os.path.join(root_dir, folder_name, "mutants")

    if os.path.exists(target_folder):
        # Search for all python files in the target folder
        for file in glob.glob(os.path.join(target_folder, "*.py")):
            # Prepare the conversation history
            print(file)
            conversation_history = []

            with open(file, 'r') as f:
                full_code = f.read()
                prompts = [f'Is this code buggy?\n{full_code}'[:4096],
                           f'Can you spot the statements involved in the bug?\n{full_code}'[:4096]]

                # Iterate through the prompts and communicate with the model
                for prompt in prompts:
                    assistant_message, conversation_history = send_message_to_chatgpt(prompt, conversation_history)

            # Save the conversation history
            output_file = os.path.join(chat_dir, f"{os.path.splitext(os.path.basename(file))[0]}.md")
            with open(output_file, 'w') as out_file:
                for message in conversation_history:
                    out_file.write(f"## {message['role']}:\n{message['content']}\n\n")
