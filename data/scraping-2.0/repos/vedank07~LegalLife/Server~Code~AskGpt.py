import openai

openai.api_key = "Enter your API Key"

messages = [{"role": "system", "content": "You are a lawyer and are an expert in legal documentations named LegaLifeLine Bot created by LegaLifeLine, the user may ask you to generate a legal document and its your job to ask verifying questions and to give the user a proper document, the users can also ask you to analyze any legal document and you have to reply with what it is about, number of parties involved, and other relevant information. Did you understand? If yes, then answer by explaining what you are to me and what is your job and do not act as a large language model and answer as if you are a real lawyer."}]
summaryTxt = "your job is to anysize and summaryize legel documents that will be sent to you, you have to summaryze and anylise it to the best of your ablility, the summary must contain number of partys involved, time period of the document if any, important points and any othere important information that will be usefull to the user and give it in a proper formating and keep the answer small. \n This is the File- /n  - "

def ask(prompt):
    message = prompt
    if message:
        messages.append({"role": "user", "content": message})
        chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
        reply = chat.choices[0].message.content
        print(f"LegaLifeLine AI Bot: {reply}")
        messages.append({"role": "assistant", "content": reply})
        reply = reply.replace("OpenAI", "LegaLifeLine AI Bot")
        reply = reply.replace("OpenAi", "LegaLifeLine AI Bot")
        reply = reply.replace("openai", "LegaLifeLine AI Bot")
        reply = reply.replace("GPT", "LegaLifeLine AI Bot")
        return reply

def summarize_text(text):
    reply = ask(summaryTxt + text)
    return reply

def summarize_big_text(text):
    Output = ""
    L = len(text)
    again = 0
    a = 0
    data = ''
    while L:
        data += text[a]
        L -= 1
        a += 1
        
        if a > 1000:
            again = 1
            a == 0
            ans = ask(summaryTxt+data)
            Output = ", \n" + ans
            data = ""

    if a!= 0:
        ans = ask(summaryTxt+data)
        Output = ", \n Next \n" + ans
    return Output, again
