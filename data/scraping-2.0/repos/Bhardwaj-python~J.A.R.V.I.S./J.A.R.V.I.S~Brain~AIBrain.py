import openai

fileopen = open("D:\\Bhardwaj\\J.A.R.V.I.S\\Data\\Api.txt")
API = fileopen.read()
fileopen.close()

def ReplyBrain(question, chat_log=None):
    file_path = "D:\\Bhardwaj\\J.A.R.V.I.S\\Database\\chat_log.txt"
    with open(file_path, "r") as file:
        chat_log_template = file.read()

    if chat_log is None:
        chat_log = chat_log_template


    openai.api_key = API

    prompt = f'{chat_log} You : {question}\nJ.A.R.V.I.S. : '
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.5,
        max_tokens = 4008,
        top_p=0.3,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    answer = response.choices[0].text.strip()

    return answer
