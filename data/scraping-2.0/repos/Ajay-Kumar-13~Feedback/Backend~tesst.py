import os
import openai
class ChatApp:
    def __init__(self):
        # Setting the API key to use the OpenAI API
        openai.api_key = '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'
        self.messages = [
            {"role": "system", "content": "you are an effective HR executive capable of seeking specific actionable feedback from employees by asking them questions and asking follow-up questions when employee answers are too vague."},
            {"role": "assistant", "content": "Do you have any feedback to give"},
        {"role": "user", "content": "My teammates are holding me back from completing my deliverables"},
        {"role": "assistant", "content": "I am sorry to hear that, I will let the manager know about it. Could you please elaborate on the specific instances when your teammates held you back?"},
        {"role": "user", "content": "Sure, they are unavailable most of the time."},
        {"role": "assistant", "content": "Thank you so much for bringing this to our attention. This matter will be addressed at the earliest. Is there anything else you would like to add?"},
        {"role": "user", "content": "Now start a new conversation by asking them their name, ensure you get the name. Start taking their feedback next by asking them if they have any problem at work. Remember ask them one question at a time."}
        ]

    def chat(self, message):
        self.messages.append({"role": "user", "content": message})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=self.messages
        )
        self.messages.append({"role": "assistant", "content": response["choices"][0]["message"].content})
        return response['choices'][0]['message']['content']
    
    def Savereturn(self) :
      return self.messages
ch = ChatApp()
msg = " "
a = ch.chat(msg)
print(a)
while True :
    msg = input('[User] ')
    if msg.lower() == "quit" :
          break
    a = ch.chat(msg)
    print(a)
    
b = ch.Savereturn()
print(*b[6::])
