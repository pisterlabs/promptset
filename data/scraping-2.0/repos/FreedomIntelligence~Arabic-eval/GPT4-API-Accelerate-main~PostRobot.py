import openai


class PostRobot:
    def __init__(self):
        self.api_key = None
        self.proxy = None
        self.model_name = "gpt-3.5-turbo"
        self.role = None
        self.base_url = "https://api.chatanywhere.cn/v1"

    def set_role(self, role):
        if self.role is None:
            self.role = {
                "role": "system",
                "content": role,
            }

    def set_thinking_engine(self, openai_key=None, proxy=None):
        self.set_openai_key(openai_key)
        self.set_proxy(proxy)

    def set_openai_key(self, apikey):
        self.api_key = apikey

    def set_proxy(self, proxy):
        self.proxy = proxy

    def request_chatgpt(self, parameters):
        openai.api_key = self.api_key
        openai.api_base = self.base_url
        response = openai.ChatCompletion.create(
            model=parameters["model"],
            messages=parameters["messages"],
            stream=parameters["stream"]  # this time, we set stream=True
        )
        content = ""
        for chunk in response:
            try:
                content += chunk['choices'][0]['delta']["content"]
            except:
                pass
        return content

    def get_prompt(self, sample):
        text = ""
        if len(sample["instruction"]) > 0 and len(sample["input"]) > 0:
            text = sample["instruction"] + "\n" + sample["input"]
        elif len(sample["instruction"]) > 0 and len(sample["input"]) == 0:
            text = sample["instruction"]
        elif len(sample["instruction"]) == 0 and len(sample["input"]) > 0:
            text = sample["input"]
        return text

    def generate(self, new_message):
        messages = []
        if self.role is not None:
            messages.append(self.role)
        messages.append({"role": "user", "content": new_message})
        parameters = {
            "model": self.model_name,
            "messages": messages,
            "stream": True
        }
        response = self.request_chatgpt(parameters)
        return True, response
