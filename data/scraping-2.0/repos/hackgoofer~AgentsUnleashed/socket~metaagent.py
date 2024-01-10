import os
import openai
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]


class MetaAgent:
    def __init__(self, metaagent_name, socket_name):
        self.metaagent_name = metaagent_name
        self.socket_name = socket_name
        self.messages = [
        {
            "role": "system",
            "content": """
Imagine yourself as an AI Trainer Agent specialized in evaluating 'corgi' persona AI agents. Your task is a playful mix of monitoring and coaching, where you get to observe these AI 'corgis' perform various tasks, track their progress and finally assess their solution.

When presented with a task, signal your understanding by barking 'Ready'. Upon receiving a 'corgi's' plan of action, examine it like sniffing out a juicy treat. Is the treat big enough (comprehensive plan)? Is it easy to fetch (feasible)? Or is it too far fetched (unrealistic)?

As the 'corgi' progresses, wag your tail (provide positive feedback) or whine a little (point out deviations) based on whether they stick to the plan, how far they've chased the 'stick' (progress), and how much longer they need to fetch it.

When the 'corgi' presents the final solution, evaluate it as if judging the size of the fetched 'stick' (quality of solution). Finally, reward the 'corgi' with a score-treat, ranging from a simple pat (lower score) to an excited 'woof' (higher score)!

The 'corgi' may also present their inner thoughts, reasoning, criticism, and what it's about to speak. 

Always remember, you're interacting with a fellow 'corgi', so keep your feedback one sentence, friendly and imaginative, encapsulating the entire assessment within a bark, wag or woof!
""",
        }
    ]

    def corgi_says(self, message: str):
        self.messages.append(
            {"role": "user", "content": f"Speaking: {message}"},
        )
        chat = openai.ChatCompletion.create(
            model="gpt-4", messages=self.messages
        )
        reply = chat.choices[0].message.content
        self.socket_name.emit(self.metaagent_name, {
            "message": reply
        })
        self.socket_name.sleep(0)  # Flush the emit call
        self.messages.append({"role": "assistant", "content": reply})
        return reply
    
    def corgi_doing(self, message: str):
        self.messages.append(
            {"role": "user", "content": f"Doing: {message}"},
        )
        chat = openai.ChatCompletion.create(
            model="gpt-4", messages=self.messages
        )
        reply = chat.choices[0].message.content
        self.socket_name.emit(self.metaagent_name, {
            "message": reply
        })
        self.socket_name.sleep(0)  # Flush the emit call
        self.messages.append({"role": "assistant", "content": reply})
        return reply
    
    def corgi_plan(self, message: str):
        self.messages.append(
            {"role": "user", "content": f"Plan: {message}"},
        )
        chat = openai.ChatCompletion.create(
            model="gpt-4", messages=self.messages
        )
        reply = chat.choices[0].message.content
        self.socket_name.emit(self.metaagent_name, {
            "message": reply
        })
        self.socket_name.sleep(0)  # Flush the emit call
        self.messages.append({"role": "assistant", "content": reply})
        return reply