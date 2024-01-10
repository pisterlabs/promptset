from datetime import datetime
import openai
# openai.api_key = "OPENAI_API_KEY"

def query(role, content):
    return {"role": role, "content": content}

def generate_resp(history, sender_email, owner_email, free_time, assistant_email):
    # Initialize the conversation with the assistant's role and a prompt
    today = datetime.now()
    formatted_date = today.strftime("%Y-%m-%d")

    # Set the context for the assistant
    body_context = f'''You are a meeting scheduling agent acting as a bridge . Your name is Zen ({assistant_email}), virtual assistant, working under {owner_email} (the owner). You will have access to the owner's availability. You can either suggest 3-4 slots for the meeting to the client, or generate a confirmation message for the client, based on the conversation provided.'''

    body_prompt = '''You are Zen ({}). Today is {}. You are working under, {} (the owner), who has free time slots: {}. 
                    They are trying to schedule a meeting with the client, {} (extract the name properly to write in email). 
                    Look at this conversation conversation: "{}". Now, based on this conversation, you have to do one of the following:
                    1) Suggest free times for the meeting from available time slots that the client can choose from. Limit the duration to 30 minutes, and provide start + end time both and the time-zone. Be biased towards suggesting times in the next 2-3 days.
                    2) Generate a confirmation message for the client containing necessary details regarding time of meeting, like starting and ending time. Do not include location of meeting.
                    Decide just one of these and then frame the body of the email, and check the time zone properly as provided.
                    Do not write subject of the email.
                    I will be sending your response directly to the client as it is, so you need to reply JUST THE BODY of email which will be automatically sent to the client. 
                    Do not even share any notes/instructions/your thought process.
                    Be very brief and to the point. Do not write anything unnecessary.
                    If you have already provided suggestions, and client is agreeing to one of them, just generate a confirmation message.
                    While giving regards/salutation, use- Zen, Virtual Assistant to "owner's name".'''.format(assistant_email, str(formatted_date), owner_email, free_time, sender_email, history) 
    messages_list = [
        query("system", body_context),
        query("user", body_prompt)
    ]
    print("SYSTEM MESSAGE: {}".format(body_context))
    print("USER PROMPT: {}".format(body_prompt))
    # Generate a response using GPT-3.5-turbo
    print("GENERATING BODY")
    try:
        body_response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages_list
        )
    except Exception as e:
        print("ERROR: {}".format(e))
        return "ERROR", "ERROR"
    print("GOT IT")

    # Extract the assistant's reply from the response
    body_reply = body_response['choices'][0]['message']['content']

    subject_context = '''I will give you an email, and generate just a single proper subject for the email. Return the subject and nothing else'''

    # Create the prompt
    subject_prompt = '''Content of the email: "{}"'''.format(body_reply)

    messages_list = [
        query("system", subject_context),
        query("user", subject_prompt)
    ]

    # Generate a response using GPT-3.5-turbo
    print("GENERATING SUBJECT")
    subject_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages_list
    )
    print("GOT IT")

    # Extract the assistant's reply from the response
    subject_reply = subject_response['choices'][0]['message']['content']

    return body_reply, subject_reply
