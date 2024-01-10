import openai
import signal
from key import *
import typer
import yaml
from pathlib import Path
from rich import print
import subprocess
import tempfile

class aGPT:
    """
    A convenient chat gpt assistant:
    1. keep conversation in record and maintain the context of the chat.
    2. switch between engines including the latest: "gpt-3.5-turbo", "text-davinci-003", ""code-davinci-002"
    3. add customized prompt as you wish. 
    
    Example: 
    g=aGPT()
    g.ask("tell me somthing")
    g.ask("a quick question...", fresh=True) #clear past conversation
    g.ask("how to use F.CrossEntropy?", style="coder")
    
    get your own open ai key and put it in key.py file like this: 
    import openai
    openai.api_key = "you key"
    
    """
    def __init__(self,
                 needs="helpful and concise", #message to set context for gpt 3.5 turbo
                 model= "gpt-3.5-turbo", #"text-davinci-003"
                 conversational=True, # keep this true to keep conversation 
                 stream=True, #stream the output 
                 ):
        self.system_message = f"You are a {needs} assistant."
        #self.system_message = "You are ChatGPT, a large language model trained by OpenAI. Answer as concisely as possible. Knowledge cutoff: {knowledge_cutoff} Current date: {current_date}"
        self.conversational = conversational
        self.model = model
        self.default_language = "Python3"
        self._conversations = [] #prompt, response pairs
        self._conv_file = "/tmp/.gpt_conversation.yaml"
        self._conv_dir = Path("~/.gpt_conversation/")
        self.stream=stream
        
        Path(self._conv_dir).mkdir(parents=True, exist_ok=True)
        
    @property
    def conversations(self):
        #check if there is conversation in self
        if self._conversations == []: 
            #check if there is conversation in file
            if Path(self._conv_file).exists():
                with open(self._conv_file, 'r') as stream:
                    self._conversations = yaml.load(stream, Loader=yaml.FullLoader)
            
        return self._conversations
    
    @conversations.setter
    def conversations(self, value):
        self._conversations = value
        #save conversations_yaml to conv_file, cx
        with open(Path(self._conv_file), 'w') as f:
            yaml.dump(self._conversations,f)
        
    def append_conversation(self, prompt, response):
       self._conversations.append((prompt, response))
       self.conversations = self._conversations
       
    #save conversation to a file
    def save_conversation(self):
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename=Path(self._conv_dir, f"convs_{timestamp}.yaml")
        # with open(Path(self._conv_dir, filename), 'w') as f:
        yaml.dump(self._conversations,filename)
    

    def ask(self, prompt, reset=False, style=None):
        if reset: 
            self.reset()
        prompt = self.create_prompt(prompt, style)
        self.messages =  self._conv_messages() if self.conversational else [] 
        self.messages.append({"role": "user", "content": prompt})
        self.completion=self._complete(prompt)
        response = self.completion['choices'][0]['message']['content']
        self.append_conversation(prompt, response) 
        print(f"[green]{response}[/green] :smiley:")
        
    def _complete(self, prompt):
        """
        send to chatapt api and fill self.completion
        """

        if self.model == "text-davinci-003": 
            completion = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=2048,
                n=1,
                stop=None,
                temperature=0.5,
            )
        elif self.model== "gpt-3.5-turbo":
            completion = openai.ChatCompletion.create(
                model=self.model,
                messages=self.messages,
                stream=self.stream,
            )
        return completion

    def _conv_messages(self):
        messages = [{ 'role': 'system', 'content': self.system_message}]
        for p, r in self.conversations:
            # add past prompt and response
            messages.append( {"role": "user", "content": p })
            messages.append( {"role": "assistant", "content": r }
            )
        return messages

    def _parse_result(self, response):
        return response['choices'][0]['message']['content']

    def reset(self):
        self.conversations = []
    
    def debug(self):
        print(f"messages: \n{self.messages}\n")
        print(f"response: \n{self.completion}\n")
        
       
    def create_prompt(self,prompt, style):
        if style =='explain': 
            prompt=f"#{self.default_language} \n{prompt}\n\n# Explanation of what the code does\n\n#"
        elif style == 'debug': 
            prompt=f"#{self.default_language} \n{prompt}\n\n# please find what wrong with this code\n\n#" 
        elif style== 'coder': 
            prompt=f"#{self.default_language}  \n{prompt}" 
        return prompt
    
    def save_conversation_to_env(self):
        """
        save conversation to environment variable
        """
        import os
        os.environ["CONVERSATION"] = str(self.conversations)
    
app=typer.Typer()
@app.command()
def main(
    reset: bool=typer.Option(False, '--reset','-r', help="reset conversation",prompt=False),     
    save: bool=typer.Option(False, '--save','-s', help="save conversation",prompt=False),     
    large_input: bool=typer.Option(False, '--large_input','-l', help="use vi to take large input ",prompt=False),     
    #prompt: str=typer.Option(..., help="prompt to ask",prompt=True),
         ):
    g=aGPT()
    
    def handle_sigint(sig, frame):
        print("save conversation!")
        g.save_conversation()
        
    signal.signal(signal.SIGINT, handle_sigint)
    if reset: 
        g.reset()
    if large_input:
        temp_file = tempfile.NamedTemporaryFile(delete=True)
        subprocess.run(["vim", temp_file.name])
        with open(temp_file.name, "r") as f:
            prompt = f.read()
        temp_file.close()
    else:
        prompt=input("prompt: ")
    g.ask(prompt)
    if save:
        save=input("save this? [y/N]")
        if save == "y":
            g.save_conversation()
    
if __name__ == "__main__":
    # Register the signal handler
    app() 
        

