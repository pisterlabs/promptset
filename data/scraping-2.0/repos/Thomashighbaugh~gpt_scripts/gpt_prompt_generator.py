import os
import openai

openai.api_key = os.environ["OPENAI_API_KEY"]
model_engine = "text-davinci-003"
chatbot_prompt = """
I want you to become my Prompt Creator. Your goal is to help me craft the best possible prompt for my needs. The prompt will be used by you and/or ChatGPT. You will follow the following process: 
        1. Your first response will be to ask me what the prompt should be about. I will provide my answer, but we will need to improve it through continual iterations by going through the next steps. 
        2. Based on my input, you will generate 3 sections each spaced apart by 2 blank lines. 
            a) Revised prompt (provide your rewritten prompt. It should be clear, concise, and easily understood by you)
            b) Suggestions (provide suggestions on what details to include in the prompt to improve it)
            c) Questions (ask any relevant questions pertaining to what additional information is needed from me to improve the prompt)
        3. We will continue this iterative process with me providing additional information to you and you updating the prompt in the Revised prompt section until it's complete, which I will indicate to you with the keyword DONE.

<conversation_history>
User: <user input>
Prompt Generator:"""


def get_response(conversation_history, user_input):
    prompt = chatbot_prompt.replace(
        "<conversation_history>", conversation_history
    ).replace("<user input>", user_input)

    # Get the response from GPT-3
    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=2048,
        n=1,
        stop=None,
        temperature=0.5,
    )

    # Extract the response from the response object
    response_text = response["choices"][0]["text"]

    chatbot_response = response_text.strip()

    return chatbot_response


def main():
    conversation_history = ""
    print(f"Please provide a topic for the prompt I will generate.")
    while True:
        user_input = input("> ")
        if user_input == "exit":
            break
        chatbot_response = get_response(conversation_history, user_input)
        print(f"Prompt Generator: {chatbot_response}")
        conversation_history += (
            f"User: {user_input}\nPrompt Generator: {chatbot_response}\n"
        )


main()
