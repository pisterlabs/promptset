import openai

openai.api_key='sk-vJGGvjcIric8ZJkAIT3BlbkFv9DWWzzM3pUQCDv'

def changeApiKey(newKey: str):
    openai.api_key=newKey

class chatgptConnect():
    def __init__(self, model: str="text-davinci-003", max_token: int=1024, temperature: float=0.5) -> None:
        self.model=model
        self.max_token=max_token
        self.temperature=temperature
    
    def reply(self, input: str, role: str='user') -> str: 
        try: 
            if len(input)>0: 
                response = openai.Completion.create(
                    model=self.model,
                    prompt=input,
                    temperature=self.temperature,
                    max_tokens=self.max_token,
                )
                return response.choices[0].text
            return ' '
        except: 
            return ' '

    def changeMaxToken(self, max_token: int):
        if max_token<=2048 and max_token>10: 
            self.max_token=max_token
            print('changeMaxToken: {}'.format(self.max_token))
        else: 
            print('change data out of range')
    
    def changeTemperature(self, temperature: float):
        if temperature<=2. and temperature>=0.:
            self.temperature=temperature
            print('changeTemperature: {}'.format(self.temperature))
        else: 
            print('change data out of range')

class chatgptContinueConnect(): 
    def __init__(self, model: str="gpt-3.5-turbo") -> None:
        self.model=model
        self.textGroup=[]
        self.total_token=0
    
    def reply(self, input: str, role: str="user") -> str: 
        overText=[]
        self.textGroup.append({"role": role, "content": input})
        message_sent=self.textGroup.copy()
        self.total_token+=len(input)
        now_total_token=self.total_token
        if now_total_token>4096: 
            while(now_total_token>4096): 
                now_total_token-=len(message_sent[0]["content"])
                overText.append(message_sent[0])
                message_sent.pop(0)
        try: 
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=message_sent
            )
            self.textGroup=message_sent.copy()
            self.total_token=now_total_token
            self.textGroup.append(response.choices[0].message)
            return response.choices[0].message['content']
        except: 
            self.textGroup.pop()
            self.total_token-=len(input)
            return ' '
