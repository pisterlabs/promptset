import openai, os

openai.api_key = os.environ.get('OPEN_AI_KEY')

def ask_gpt3(chat_history, model="gpt-3.5-turbo", temperature=None, top_p=None, system_message=None):
    messages = []
    
    # If there's a system message, prepend it to the list
    if system_message:
        messages.append({"role": "system", "content": system_message})
    
    # Extend with the chat history
    messages.extend(chat_history)
    #print(f"This is the message {messages}")

    payload = {
        "model": model,  # Or make the model dynamic based on config
        "messages": messages,
        "max_tokens": 1500
    }
    
    if temperature:
        payload["temperature"] = temperature
    if top_p:
        payload["top_p"] = top_p
    
    response = openai.ChatCompletion.create(**payload)
    return response.choices[0].message.content.strip()

def summarize_with_gpt3(chat_history, model="gpt-3.5-turbo", system_message="Ovanstående var en konversation mellan en användare och en AI. Sammanfatta konversationen med de viktigaste punkterna:", temperature=0.7, top_p=0.9):
    """
    Gets a summary of a given chat history using OpenAI.
    """
    # Convert the chat history text into a list of messages
    messages_list = [{"role": "user", "content": message} for message in chat_history.split("\n") if message]
    #print(f"This is the messages list sent {messages_list}")
    
    messages_list.append({"role": "user", "content": system_message})

    # Get a response from OpenAI
    response = ask_gpt3(messages_list, model, temperature, top_p)
    #print(f"This is the response from OpenAI {response}")
    
    return response


def generate_reminder_from_summary(summary, model="gpt-3.5-turbo", system_message="Based on this summary of a previous session, remind the user what you last spoke about and ask about it.", temperature=0.7, top_p=0.9):
    """
    Uses the provided summary to generate a reminder message for the user.
    """
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": summary}
    ]
    
    response = ask_gpt3(messages, model, temperature, top_p, None)
    
    return response
