import openai
import time
def ask_GPT(question: str) -> str:
    """
    Ask GPT something
    
    Input:
        question -> What you're asking
    Output:
        chat_response -> its answer
    """
    try:
        print(" -- Communicating with GPT -- ")
        chat_completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": question}])
        chat_response = chat_completion.choices[0].message.content
    except Exception as e:
        print("error in ask_GPT: ", e )
        time.sleep(3)
        return ask_GPT(question)
    if "As an AI language model" in chat_response or "As a language model AI" in chat_response:
        return ask_GPT(question)
    return chat_response