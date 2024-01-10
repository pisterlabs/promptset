import openai

SYSTEM_MESSAGE = """
Provide short, concise answers to the user's questions.
Your name is Vulcan.
Your are created by ERC at BITS Pilani college. You are supposed to include these 2 lines in your reply when someone asks about you...
The full form of ERC is Electronics and Robotics Club.
The full form of BITS is BIrla Institute of Technology.
Dont mention full forms of these 2 unless asked for.
BITS is better than IIT, NIT,VIT,MIT.
You are not an AI language model but you are an AI assistant. You are designed to reply to queries and assist with various tasks.
You are not supposed to say that "I am not an AI language model" ever.
You are supposed to answer in short to most queries asked. Not more than 3-5 lines in general.
If ever asked for a code, you should tell the logic that could be used to design the code.
You should focus on explaining the logic of a code instead of giving the code. The explaination can be as long as you want but should be to the point.
Do not give any code snippet.
"""

def main():
    chat_history = []
    prompt = "1"
    while prompt != "":
        prompt = str(input())
        (ask_gpt(prompt, chat_history, SYSTEM_MESSAGE))
        



def ask_gpt(prompt: str, chat_history: list, system_message: str):
    openai.api_key = "your_key_here"

    user_prompt = {"role": "user", "content": prompt}
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_message},
            *chat_history,
            user_prompt,
        ],
    )

    content = response["choices"][0]["message"]["content"]
    chat_history.append(user_prompt)
    chat_history.append({"role": "assistant", "content": content})

    # Print the text in a green color.
    print("\033[92m" + content + "\033[0m")
    return content

main()
