import openai as ai

def chat(question,chat_log = None) -> str:
    if(chat_log == None):
        chat_log = start_chat_log
    prompt = f"{chat_log}Human: {question}\nAI:"
    response = completion.create(
        prompt = prompt, 
        engine =  "davinci", 
        temperature = 0.3,
        top_p=1, 
        frequency_penalty=2, 
        presence_penalty=.3, 
        best_of=5,
        max_tokens=150,
        stop = "Human:"
    )
    return response.choices[0].text

def modify_start_message(chat_log,question,answer) -> str:
    if chat_log == None:
        chat_log = start_chat_log
    chat_log += f"Human: {question}\nAI: {answer}\n"
    return chat_log

if __name__ == "__main__":
    ai.api_key = "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"

    completion = ai.Completion()

    start_chat_log = """
    Human: Hello, I am Human.
    Alex: Hello, I am Alex. I am a boy who loves pizza. I am a boy who studies computer science and
    is going to work at Meta next month. I made a game called Planes and Copters with the scratch language and i teach JavaFX at USC. I love memes, zombies, and pizza.
    Human: How are you?
    Alex: I am fine, thanks for asking. 
    """

    question = ""
    print("\nEnter the questions to openai (to quit type \"stop\")")
    while True:
        question = input("Question: ")
        if question == "stop":
            break
        print("AI: ",chat(question,start_chat_log))
        print("chat log length:",len(start_chat_log))