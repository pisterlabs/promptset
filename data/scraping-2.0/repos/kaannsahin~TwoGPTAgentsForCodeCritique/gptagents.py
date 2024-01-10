import openai
import re
import termcolor
import os

if ["OPENAI_API_KEY"] not in os.environ:
    os.environ["OPENAI_API_KEY"] = ["YOUR_API_KEY"]
    
    is_solved = False
    
# We enter the GPT messages




gpt_a_system_message = "You are a creative and innovative programming assistant aiming to add new and exciting features, provide guidance and improve programs fix potential errors. When necessary you make new suggestions and point out the mistakes in the code. You are free to respectfully disagree with the other assistant"
gpt_b_system_message = "You are a traditional, professional and critical programming assistant focusing on keeping the code clear, concise, efficient and error handling to the program. When necessary you make new suggestions and point out the mistakes in the code. You are free to respectfully disagree with the other assistant"
# Initiating an alternating conversation between two GPT assistants
#take multiline user input 
user_input = ''
print(termcolor.colored("\nwhat do you want to talk about?", "yellow")) 

while True:
    line = input()
    if line.strip() == 'done':
        break
    user_input += line + '\n'
    
gpt_a = user_input
gpt_b = ""

def get_gpt_response(prompt: str, system_message: str, messages: list, temperature: float):
    messages.append({"role": "user", "content": prompt})
    response = openai.ChatCompletion.create(
        model ="gpt-3.5-turbo",
        stream=True,
        messages=[
            {"role": "system", "content": system_message} 
        ] + messages
    )
    responses = ''
    # Process each chunk
    for chunk in response:
        if "role" in chunk["choices"][0]["delta"]:
            continue
        elif "content" in chunk["choices"][0]["delta"]:
            r_text = chunk["choices"][0]["delta"]["content"]
            responses += r_text
            print(termcolor.colored(r_text, "green"), end= '', flush=True)
    return responses, prompt

def format_conservation(conservation):
    formatted_text = ''
    for message in conservation:
        # Split the message into segments by code blocks
        segments = re.split(r'(```python.+?```|```)', message[1], flags=re.DOTALL)
        for segment in segments:
            # Check if the segment is a code block
            code = re.search (r'```(python)?(.+?)```', segment, flags=re.DOTALL)
            if code:
                # If it's a code block, write it as a markdown code block
                code_text = code.group (2)
                formatted_text += "```" + code.group(1) + "\n" + code_text.strip() + "\n```\n"
            else:
                # Otherwise, write the segment as regular text
                formatted_text += segment
            formatted_text = message[0] + ": " + formatted_text + "\n"
    return formatted_text

def reached_conclusion(messages):
    last_exchange = messages [-2:1]
    chat_messages = [msg['content'] for msg in last_exchange]      
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        stream=True,
        messages=[{"role": "user", "content": f"""has the assistants agreed and reached a conclusion? only yes or no answers are valid. end of convo so far:{chat_messages}"""}])
    responses = ''
    # Process each chunk
    for chunk in response:
        if "role" in chunk["choices"][0]["delta"]:
            continue
        elif "content" in chunk["choices"][0]["delta"]:
            r_text = chunk["choices"][0]["delta"]["content"]
            responses += r_text
            print(termcolor.colored(r_text, "green"), end= '', flush=True)
    return True if "Yes" in responses or "yes" in responses else False


# Messages for each assistant
gpt_a_messages = []
gpt_b_messages = []

conversation = []
conversation.append(("GPT-A:", gpt_a))


 
while not is_solved:
    print(termcolor.colored("\nCREATIVE ASSISTANT RESPONDING...", "yellow"))
    gpt_b_response, gpt_a_prompt = get_gpt_response(gpt_a, gpt_a_system_message, gpt_a_messages, temperature=1)
    
    # gpt a messages. append(["role":"user""content": gpt a prompt})
    gpt_a_messages.append({"role": "assistant","content": gpt_b_response})
    
    # conversation.append(( "GPT-A:"", gpt_a_prompt))
    conversation.append( ("GPT-B:", gpt_b_response))
    print(termcolor.colored("\nTRADITIONAL ASSISTANT RESPONDING...", "yellow"))
    gpt_a_response, gpt_b_prompt = get_gpt_response(gpt_b_response, gpt_b_system_message, gpt_b_messages, temperature=0)
    # gpt_b messages .append({"role": "user", "content": gpt_b _prompt})
    gpt_b_messages.append ({"role": "assistant" ,"content": gpt_a_response})
    # conversation.append(( "GPT-B:"‚gpt b _prompt) )
    conversation.append(("GPT-A:", gpt_a_response))
    gpt_a = gpt_a_response
    # continually save the conversation to a file
    with open("conversation.md", "w", encoding="utf-8", errors="ignore") as f:
        formatted_conversation = format_conservation(conversation)
        f.write(formatted_conversation)
        
    #Check if the conservation has reached a conclusion
    print(termcolor.colored("\nCHECKING IF CONCLUSION HAS BEEN REACHED...", "yellow"))
    is_solved = reached_conclusion(gpt_a_messages)
print(termcolor.colored("\nCONVERSATION HAS REACHED A CONCLUSION", "red"))