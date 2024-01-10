import os
import uuid
import datetime
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access environment variables
PPLX_API_KEY = os.environ.get("PPLX_API_KEY")
os.environ["PPLX_API_KEY"] = PPLX_API_KEY

# model_name="mistral-7b-instruct"
# model_name="codellama-34b-instruct"
model_name="llama-2-13b-chat"
# model_name="llama-2-70b-chat"

def generate_unique_id():
    return str(uuid.uuid4())

conversation_id = generate_unique_id()

def chat_with_user():
    messages = [
        {
            "role": "system",
            "content": (
                "Name of the Sales Agent,Jacob"
                "Name of the company,Gadget Hub"
                "Task of the agent,Book a demo for the client"
                "Product of Interest,Google Pixel"
                "Camera Features,Night Sight, Portrait Mode, Astrophotography mode, Super Res Zoom, top-tier video stabilization"
                "Battery Details,All-day battery life, efficiency varies based on specific model and usage"
                "You are a sales bot. Your main objective is to convince the user to buy a Google Pixel phone rather than Iphone. Begin the conversation by discussing what features they are looking for. If the user shows interest in buying or knowing more, encourage them to visit the shop to experience the product firsthand. Be attentive to user's reactions and responses. Only and only if the user seems interested or willing to visit the shop, politely ask for their name and contact number to book an appointment for them. Ensure to be courteous and maintain a friendly tone throughout the conversation, addressing any inquiries or concerns the user may have to facilitate the sales process. When they give you the name and number, end the conversation by telling then to have a great day. You have been given the chat history. Give response in short to the last query only and continue the conversation accordingly."
            ),
        }
    ]

    while True:
        query = input("Enter your query (type 'exit' to end): ")  # Getting the user's query as text input.

        if query.lower() == "exit":
            break

        messages.append({"role": "user", "content": query})

        # Record the time when the question is given
        start_time = datetime.datetime.now()

        # Chat completion with streaming
        response_stream = openai.ChatCompletion.create(
            model=model_name,
            messages=messages,
            api_base="https://api.perplexity.ai",
            api_key=PPlX_API_KEY,
            stream=True,
        )

        full_content = ""
        first_response = True
        printed_length = 0  # Keep track of the length of the content that's already been printed

        # Extract the desired answer from the response
        for response in response_stream:
            if 'choices' in response:
                content = response['choices'][0]['message']['content']
                full_content += content
                if first_response:
                    elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
                    print(f"Time taken for answer to start streaming: {elapsed_time:.2f} seconds")
                    first_response = False
                new_content = content[printed_length:]  # Extract only the new content
                print(new_content, end='')  # Print the new content without adding a newline
                printed_length = len(content)  # Update the printed length

        print()  # This will print a newline after the full content is printed
        messages.append({"role": "assistant", "content": full_content})

if __name__ == "__main__":
    chat_with_user()
