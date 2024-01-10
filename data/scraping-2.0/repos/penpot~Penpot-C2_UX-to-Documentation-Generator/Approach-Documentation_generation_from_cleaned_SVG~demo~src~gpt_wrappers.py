import openai

class ChatGPTWrapper:
    """
        A wrapper for the OpenAI Chat API that allows for a memory to be maintained
        between calls to the API. This is useful for a chatbot that is maintaining
        a conversation with a user.
    """
    def __init__(self, system_prompt, model="gpt-3.5-turbo", temperature=0.2, top_p=1):
        self.messages = []
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p

    def initialize_with_question_answer(self, question, answer):
        self.messages = [
            {
                "role": "system",
                "content": self.system_prompt
            },
            {
                "role": "user",
                "content": question
            },
            {
                "role": "assistant",
                "content": answer
            }
        ]
        
    def generate(self, question):
        self.messages.append(
            {
                "role": "user",
                "content": question
            }
        )
        response = openai.ChatCompletion.create(
            model=self.model,
            messages = self.messages,
            temperature = self.temperature,
            top_p = self.top_p
            
        )
        self.messages.append(
            {
                "role": "assistant",
                "content": response['choices'][0]['message']['content']
            }
        )
        return response['choices'][0]['message']['content']