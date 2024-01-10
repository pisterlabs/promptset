import openai
from openai.error import ServiceUnavailableError
import time

class Chatter:
    def __init__(self, 
            chat_prompt: str = "You are a virtual assistant", 
            chat_horison: int = 10, 
            chat_tokens: int = 100,
            temp: float = 0.5,
            stream = False, 
            filt_prompt: str = "Who should respond to this? Reply with 'USER', 'ASSISTANT' or 'BOTH'",
            filt_horison: int = 1, 
            filt_name: str = "asssistant",
            filt_keys: list[str] = ["ASSISTANT","BOTH"],
            filt_tokens: int = 5):
        """
        Creates a Chatter object for natural communication with ChatGPT

        Args:
            chat_prompt (str): The description of the chatters's role
            chat_horison (int): How many messages are used for the next response. 1 uses only last message. Chatter responses are counted.
            chat_tokens (int): How many tokens chatter can respond with. Will be a hard cut if reached. 
            temp (float): How varied the responses should be. 0 = deterministic, 2 = very random.
            stream (bool): If the answers should be yielded in tokens or returned as a full string.
            filt_prompt (str): The prompt for the filter which decides if a response should be given.
            filt_horison (int): How many messages are used to decide if to respond. A value <= 0 turns of filtering.
            filt_name (str): What the chatter is called when deciding to filter. The user is always called 'user'.
            filt_keys (list[str]): If any of the keys are returned by filter, chatter will respond.
            filt_tokens (int): How many tokens the filter might return. Will be a hard cut if reached. 
        """
        if not openai.api_key:
            openai.api_key = open("openai.key").read().strip()
        self.stream = stream 
        self.messages = []
        self.NLP_model = "gpt-3.5-turbo"
        self.chat_base = [ {"role": "system", "content": chat_prompt} ]
        self.chat_horison = chat_horison
        self.chat_tokens = chat_tokens
        self.temp = temp
        self.filt_base = [{
            "role": "system", 
            "content": filt_prompt
            } ]
        self.filt_horison = filt_horison
        self.filt_name = filt_name
        self.filt_keys = filt_keys
        self.filt_tokens = filt_tokens

    def get_response(self) -> str:
        """
        Return a response based on the last chat_horison messages

        Returns:
            str: The response
        """
        try:
            response = openai.ChatCompletion.create(
                model=self.NLP_model, 
                messages=self.chat_base + self.messages[-min(len(self.messages), self.chat_horison):],
                temperature=self.temp,
                max_tokens = self.chat_tokens,
                stream = False
            ).choices[0].message.content
            return response
        except ServiceUnavailableError:
            time.sleep(0.1)
            return self.get_response()

    def stream_response(self) -> "Generator[str,None,None]":
        """
        Yields a response based on the last chat_horison messages
        Faster than get_response

        Returns:
            Generator[str,None,None] : Yields the tokenised response
        """
        try:
            for chunk in openai.ChatCompletion.create(
                model=self.NLP_model, 
                messages=self.chat_base + self.messages[-min(len(self.messages), self.chat_horison):],
                temperature=self.temp,
                max_tokens = self.chat_tokens,
                stream = True
            ):
                yield chunk.choices[0].delta.get("content","")
        except ServiceUnavailableError:
            time.sleep(0.1)
            return self.stream_response()
    
    def should_respond(self) -> bool:
        """
        Concludes if Chatter should reply to the last received message.
        Does this via evaluating with filt_prompt. Should respond if any filt_key is returned. 
        Based on the last filt_horison messages. Setting filt_horison <= 0 turns of filtering

        Returns:
            bool : Wheter to respond or not
        """
        try:
            if(self.filt_horison <= 0): return True
            joined_messages = "\n ".join(
                [(m["role"] if m["role"] != "assistant" else self.name) + ": " + m["content"] for m in self.messages[-min(len(self.messages), self.filt_horison):]]
            )
            response = openai.ChatCompletion.create(
                model=self.NLP_model, 
                messages=self.filt_base + [{"role": "user", "content": joined_messages}],
                temperature=0,
                max_tokens=self.filt_tokens
            ).choices[0].message.content
            return any([key.upper() in response.upper() for key in self.filt_keys])
        except ServiceUnavailableError:
            time.sleep(0.1)
            return self.should_respond()
    
    def __call__(self, message: str) -> str:
        
        """
        Returns a reply to message based on the last chat_horison messages
        if it concludes it should based on the last filt_horison messages

        Returns one of:
            str/Generator[str,None,None] : The response or a genereator thereof if stream is true. Empty if no response
        """
        try:
            self.messages.append(
                {"role": "user", "content": message},
            )

            if not self.should_respond():
                return "" if not self.stream else []
            
            if not self.stream:
                response = self.get_response()
                self.messages.append(
                    {"role": "assistant", "content": response},
                )
                return response
            
            def res(self): # A hack to alternate return between generator and classic returns
                response = ""
                for chunk in self.stream_response():
                    response += chunk
                    yield chunk
                self.messages.append({"role": "assistant", "content": response})
            return res(self)
        except KeyError as KE: # In case the model is overloaded
            return f"{KE}"
                
if __name__ == "__main__":
    name = "Pepper"
    f"You will be given a conversation between a group of humans, user, and a robot called {name}. Your task is to conclude if the last message from the humans is directed at {name} or to themselves. If it's directed at the humans respond only with 'HUMANS'. If it's directed at {name} respond only with '{name.upper()}'. If it could be answered by either, respond with 'BOTH'."
    swe_intro = f"Du är den mänskliga roboten {name}. En NAO-modellrobot byggd av Softbank och programmerad av FIA Robotics. Din uppgift är att hålla en intressant konversation med en grupp människor. Du får max svara med två meningar."
    eng_intro = f"You are the humanoid robot {name}. A NAO model robot built by Softbank and programmed by FIA Robotics. Your task is to hold an interesting conversation with a group of humans. You can at most answer with two sentences"
    stream_bot = Chatter(base_prompt=swe_intro,stream=True,name=name)
    chunk_bot = Chatter(base_prompt=swe_intro,name=name)
    while True:
        message = input("User: ")
        # Stream bot first
        print(f"StreamBot: ",end="")
        for i in stream_bot(message):
            print(i, end="")
        print()
        # Chunk bot second
        print(f"ChunkBot: {chunk_bot(message)}")