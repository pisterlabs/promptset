
import openai

class TravelPlanner:
    
    openai.api_key = "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"

    def __init__(self, start_date: object, end_date: object, location: str):
        self.start_date = start_date
        self.end_date = end_date
        self.location = location
        system_content = self.CreateSystemInput()
        user_input = self.CreateUserInput()
        self.response = self.OpenAIAPI(content=system_content, user_input=user_input)
    
    def OpenAIAPI(self, content: str, user_input: str) -> str:   
        response = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": content}, 
                {"role": "user", "content": user_input}
            ]
        )
        return response

    def CreateSystemInput(self):
        system_content = "You are a helpful assistant"
        return system_content

    def CreateUserInput(self) -> str:
        user_input = f"I want to travel from {self.start_date} to {self.end_date} to {self.location}. Could you make a schedule for me?"
        return user_input
    
    def GetTravelPlan(self) -> str:
        return print(self.response['choices'][0]['message']['content'])
    
    def GetRequestCost(self) -> str:
        promt_tokens = self.response['usage']['prompt_tokens']
        completion_tokens = self.response['usage']['completion_tokens']
        promt_price = 0.0015  # /1k tokens
        completion_price = 0.002  # /1k tokens
        prompt_cost = promt_tokens * promt_price / 1000
        completion_cost = completion_tokens * completion_price / 1000
        total_cost = prompt_cost + completion_cost
        return print(f"Cost of request: ${total_cost:.6f}")

# TravelPlanner(start_date="2021-10-01", end_date="2021-10-02", location="New York").GetTravelPlan()
