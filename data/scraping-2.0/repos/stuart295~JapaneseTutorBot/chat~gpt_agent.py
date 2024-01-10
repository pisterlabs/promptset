import openai

from chat.base_agent import BaseAgent


class GptAgent(BaseAgent):

    def __init__(self, agent_name: str = None, instruction_prompt=None, gpt_model="gpt-4"):
        super().__init__()
        self.agent_name = agent_name
        self.gpt_model = gpt_model
        self.messages = []

        if instruction_prompt:
            self.messages.append({"role": "system", "content": instruction_prompt})

    def tell(self, message, speaker_name: str = None, role="user"):
        payload = {"role": role, "content": message}
        if speaker_name:
            payload["name"] = speaker_name
        self.messages.append(payload)

    def listen(self, temperature=0.7) -> str:
        response = openai.ChatCompletion.create(
            model=self.gpt_model,
            messages=self.messages,
            temperature=temperature,
        )

        self.messages.append(response.choices[0]["message"])
        return response.choices[0]["message"]["content"].strip()

    def rewind(self, to_message_idx: int):
        self.messages = self.messages[:to_message_idx+1]
