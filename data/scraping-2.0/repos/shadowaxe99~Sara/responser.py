from datetime import datetime
import openai

def query(role, content):
    return {"role": role, "content": content}

def generate_resp(email_body, email_address, to_address, free_time, to_sender=True, cc=False):
    # Initialize the conversation with the assistant's role and a prompt
    today = datetime.now()
    formatted_date = today.strftime("%Y-%m-%d")

    # Determine the prompt based on 'to_sender' and 'cc'
    if to_sender and cc:
        prompt = f"YOU ARE A MEETING SCHEDULER. MAKE A RESPONSE FOR THIS CC'd EMAIL- Email from {email_address} to {to_address}:\n{email_body}\n\n"
    elif to_sender and not cc:
        prompt = f"YOU ARE A MEETING SCHEDULER. MAKE A RESPONSE FOR THIS DIRECT EMAIL TO SENDER- Email from {email_address}:\n{email_body}\n\n"
    elif not to_sender and cc:
        prompt = f"YOU ARE A MEETING SCHEDULER. MAKE A RESPONSE FOR THIS CC'd EMAIL TO RECEIVER- Email from {email_address} to {to_address}:\n{email_body}\n\n"
    else:
        prompt = f"YOU ARE A MEETING SCHEDULER. MAKE A RESPONSE FOR THIS DIRECT EMAIL TO RECEIVER- Email from {email_address}:\n{email_body}\n\n"

    messages_list = [
        query("system", "You are a helpful assistant."),
        query("user", prompt),
        query("system", f"Today's date is {formatted_date}. Free time: {free_time}")
    ]

    # Generate a response using GPT-3.5-turbo
    print("GENERATING RESPONSE")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages_list
    )
    print("GOT IT")

    # Extract the assistant's reply from the response
    assistant_reply = response['choices'][0]['message']['content']

    return assistant_reply
