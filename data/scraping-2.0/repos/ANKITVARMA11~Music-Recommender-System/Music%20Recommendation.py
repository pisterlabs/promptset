import openai

openai.api_key = "  "

def chat_with_gpt(prompt, max_tokens= 30):
    response = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages =[
            {"role": "system", "content": "recommed music according to moods"}, 
            {"role":"user", "content": prompt}
            ],
        max_tokens= max_tokens
    )

    return response.choices[0].message.content.strip()
if __name__ == "_main_":
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "bye", "exit", "stop"]:
            print("Goodbye! If you have more questions, feel free to ask.")
            break

        response = chat_with_gpt(user_input)
        print("Musicq: ",response)
