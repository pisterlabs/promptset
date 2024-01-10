import openai

openai.api_key = "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"

categories = {
    0: "classes at university", 
    1: "work on their social life at university", 
    2: "applying to and finding internships and jobs from university", 
    3: "possible roomate problems at univeristy", 
    4: "possible romantic or social relationship issues", 
    5: "their situation", 
}

system_msg = int(input("How would you like me to respond to you? "))
therapist = "You are an empathetic, understanding, and friendly therapist helping someone with imposter syndrome with " + categories[system_msg]
messages = [{"role": "system", "content": therapist}]
print("Your new assistant is ready!")

while True:
    user_message = input("You: ")
    if user_message.lower() == 'quit()':
        break
    
    # Append user message to the messages
    messages.append({"role": "user", "content": user_message})
    
    # Query the mode
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=100,
        temperature=0.8
    )
    
    # Get model's message
    assistant_message = response['choices'][0]['message']['content']
    # Append model's message to the messages
    messages.append({"role": "assistant", "content": assistant_message})
    
    print("Assistant: " + assistant_message)