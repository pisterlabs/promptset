import openai

# Read the OpenAI Api_key
openai.api_key = open("OpenAI_API_Key.txt", "r").read().strip()

# --------------------------------------------------------------
# Ask ChatGPT a Question
# --------------------------------------------------------------


# simple API call
chat_response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are an AI tutor that assists school students with math homework problems."},
        {"role": "user", "content": "Help me solve the equation 3x - 9 = 21."},
        {"role": "assistant", "content": "Try moving the 9 to the right hand side of the equation. What do you get?"},
        {"role": "user", "content": "3x = 12"}
    ]
)

print(chat_response)

