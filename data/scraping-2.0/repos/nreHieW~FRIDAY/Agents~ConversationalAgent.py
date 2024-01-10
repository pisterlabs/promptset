import openai
from .Agent import Agent
import time
from .PromptTemplate import PromptTemplate

class ConversationalAgent(Agent):
    '''
    Conversational agent
    '''
    def __init__(self, company_info) -> None:
        pass

    def __str__(self) -> str:
        return "Basic agent for basic conversational tasks with no access to tools"

    @property
    def prompt(self) -> PromptTemplate:
        s = "You are a customer service chatbot. You are to handle friendly chit chat and be conversational."
        return PromptTemplate.PromptTemplate(s)

    def generate_answer(self, query, chat_history = list(), system_prompt = None) -> dict:
        '''
        Generate answer to a query, chooses the right task
        '''

        if system_prompt is None:
            system_prompt = self.prompt.format()
        while True:
            try:
                messages = [{"role": "system", "content": system_prompt}] + chat_history + [{"role": "user", "content": query}]
                ans = openai.ChatCompletion.create(
                model= "gpt-3.5-turbo",
                messages = messages,
                temperature = 0,
                )
                reply = self.remove_ai_sentences(ans['choices'][0]['message']['content'])
                usage = ans["usage"]
                return {
                    "answer": reply,
                    "usage": dict(usage),
                }
            except openai.error.RateLimitError:
                print("Rate limit exceeded, waiting 15 seconds")
                time.sleep(15)
            
        
    def remove_ai_sentences(self, text):
        sentences = text.split(".")
        for i, sentence in enumerate(sentences):
            if "AI language model" in sentence:
                if i+1 < len(sentences) and sentences[i+1].startswith("However"):
                    return ".".join(sentences[:i] + sentences[i+2:])
                else:
                    return ".".join(sentences[:i] + sentences[i+1:])
        return text.replace("However,", "").strip()
