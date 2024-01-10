from dotenv import load_dotenv
import os  
import openai
load_dotenv()

# Get the API key from the .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
completion = openai.Completion()

def ReplyFrame(question, chat_log = None):

    FileLog = open("src\\database\\chat_log.txt", "r")
    chat_log_template = FileLog.read()
    FileLog.close()

    if chat_log is None:
        chat_log = chat_log_template

    prompt = f"{chat_log}You: {question}\nUltron: "

    response = completion.create(
        model = "text-davinci-003",
        prompt = prompt,
        temperature = 0.5,
        max_tokens = 60,
        top_p = 0.3,
        frequency_penalty = 0.5,
        presence_penalty = 0,
    )

    answer = response["choices"][0].text.strip()
    chat_log_template_update = chat_log_template + f"You: {question} \nUltron: {answer}\n"
    FileLog = open("src\\database\\chat_log.txt", "w")
    FileLog.write(chat_log_template_update)
    FileLog.close()

    return answer


def Testing():
    while True:
        askQuestion = input("You: ")                
        result = ReplyFrame(askQuestion)
        print(result)

