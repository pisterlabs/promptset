import openai

class ChatBot:
    def __init__(self):
        openai.api_key=''
    
    def get_response(self,user_input):
        responses = openai.Completion.create(
            engine="text-davinci-003",prompt=user_input,max_tokens=4000,temperature=0.5
        ).choices[0].text

        return responses

if __name__ == "__main__":
    chatbot = ChatBot()
    response = chatbot.get_response('how many countries in the world')
    print(response)
    

