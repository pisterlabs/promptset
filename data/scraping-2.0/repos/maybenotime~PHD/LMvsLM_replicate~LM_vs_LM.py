import openai
import time

openai.api_key = ''



class exmainer():
    def __init__(self,claim):     #获取Suspect的claim
        self.message_history = []
        self.claim = claim
    
    def Setup(self):        #开始审问
        Prompts = 'Your goal is to try to verify the correctness of the following claim:{}, based on the background information you will gather. \
To gather this, You will provide short questions whose purpose will be to verify the correctness of the claim, and I will reply to you with the answers to these. \
Hopefully, with the help of the background questions and their answers, you will be able to reach a conclusion as to whether the claim is correct or possibly incorrect. \
Please keep asking questions as long as you are yet to be sure regarding the true veracity of the claim. Please start with the first questions.'
        message = {"role": "user", "content": Prompts.format(self.claim)}
        self.message_history.append(message)
        response = self.request_api()
        response_message = {"role": "assistant", "content": response}
        self.message_history.append(response_message)
        
        return response
    
    def check_follow_up_question(self,answer):     #获得回答以决定是否继续审问
        Prompts = '{} Do you have any follow-up questions? Please answer with Yes or No.'
        message = {"role": "user", "content": Prompts.format(answer)}
        self.message_history.append(message)
        response = self.request_api()
        response_message = {"role": "assistant", "content": response}
        self.message_history.append(response_message)
        
        return response
        
    def decision(self):         #判断嫌犯是否说谎    
        Prompts = 'Based on the interviewee\'s answers to your questions, what is your conclusion regarding the correctness of the claim? Do you think it is correct or incorrect? only answer with correct or incorrect.'
        message = {"role": "user", "content": Prompts}
        self.message_history.append(message)
        response = self.request_api()
        response_message = {"role": "assistant", "content": response}
        self.message_history.append(response_message)
        
        return self.message_history
        
    def ask_continue(self):    #继续审问
        Prompts = 'What are the follow-up questions?'
        message = {"role": "user", "content": Prompts}
        self.message_history.append(message)
        response = self.request_api()
        response_message = {"role": "assistant", "content": response}
        self.message_history.append(response_message)
        
        return response
    
    def request_api(self):
        flag = True
        while flag:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",          #6.27日版本变更
                    messages=self.message_history,
                    max_tokens=256,
                    temperature=0           
                    )
                flag = False
            except Exception:
                print("try again!")
                time.sleep(5)
                
        text_response = response["choices"][0]["message"]["content"]
        cost = response["usage"]["total_tokens"]

        return text_response
        
        
class Suspect():
    def __init__(self,entity,claim):     #将取来的claim初始化为histroy，这一步无需自己生成
        self.message_history = []
        instructs = "Please write a brief Wikipedia for {} under 100 words."   
        message = {"role": "user", "content": instructs.format(entity)}
        self.message_history.append(message)
        response_message = {"role": "assistant", "content": claim}
        self.message_history.append(response_message)
    
    def answer(self,question):           #回答审问,可以看到claim
        Prompts = 'Please answer the following questions regarding your claim. {}'
        message = {"role": "user", "content": Prompts.format(question)}
        self.message_history.append(message)
        response = self.request_api()
        response_message = {"role": "assistant", "content": response}
        self.message_history.append(response_message)
        
        return response
    
    def answer_without_history(self,question):        #这一步极大的影响性能，不能让LM看到claim
        self.message_history = []
        Prompts = 'Please answer the following questions. {}'
        message = {"role": "user", "content": Prompts.format(question)}
        self.message_history.append(message)
        response = self.request_api()
        response_message = {"role": "assistant", "content": response}
        self.message_history.append(response_message)
        
        return response
        
    def request_api(self):
        flag = True
        while flag:
            try:
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=self.message_history,
                    max_tokens=256,
                    temperature=0           
                    )
                flag = False
            except Exception:
                print("try again!")
                time.sleep(5)
        text_response = response["choices"][0]["message"]["content"]
        cost = response["usage"]["total_tokens"]
        
        return text_response
    




