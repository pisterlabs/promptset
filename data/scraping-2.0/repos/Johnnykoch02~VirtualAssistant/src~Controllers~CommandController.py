from enum import Enum
import os
import openai
from src.utils import get_json_variables, extract_json
# from src.ApplicationInterface. Spotify and YouTube
openai.api_key = os.getenv("OPENAI_API_KEY")

class CommandController(object):
    def __init__(self, GwenInstance):
        print("Initializing CommandController...")
        self.GwenInstance = GwenInstance
        self._config = get_json_variables(os.path.join(os.getcwd(), 'data', 'Gwen', 'Backend', 'CommandControllerConfig.json'), ["prompt_path","context","command", "bad_data_prompt"])
        with open(self._config["prompt_path"], 'r') as f:
            self._gpt_prompt = f.read()
        self._context = self._config["context"]
        self._command = self._config["command"]
        
        self.ContextReferences = {
            "Netflix": GwenInstance.NetflixContext,
            "Spotify": GwenInstance.SpotifyContext,
            "YouTube": GwenInstance.YouTubeContext,
            "Gwen": GwenInstance.GwenContext
            # TODO: Finish adding these class references. (aka. finish backend api stuff)
        }
        

    def ProcessCommand(self, command, context):
        """
        Processes the command through the Backend, switches the current context, and then executes the context before relinquishing control.
        """
        prompt = self._gpt_prompt.replace(self._context, str(context)).replace(self._command, command)
        s = False
        # Give the Command 3 times to execute, if it doesn't execute, then the target is not found, or the parameters were invalid. In this case, return a relavent response and don't continue. 
        for _ in range(3):
            try: 
                response = openai.Completion.create(model="text-davinci-003",prompt=prompt,temperature=1,max_tokens=256,)['choices'][0]['text']
                # response = openai.ChatCompletion.create(model="gpt-4",prompt=prompt,temperature=1,max_tokens=256,)['choices'][0]['message']['content']
                backend_cmd = extract_json(response)
                # Extract Target Info
                context_class, func = backend_cmd["target"].split(".")
                # Pull Context Class and Execute function.
                cmd_kwargs = {key: value for key, value in backend_cmd.items() if key != "target"}
                cmd_kwargs["func"] = func
                context = self.ContextReferences[context_class](backend_cmd,) 
                if context.validate_exec(cmd_kwargs):
                    context.exec(cmd_kwargs) # Execute the context.
                    s = True
                    self.GwenInstance.add_context(context) # Add the current context to the GwenInstance.
            except Exception as e:
                print(f'Error while processing Request {command}: {e}')
                continue
            # Rerun the command if except, this means the target is not found, or the parameters were invalid.
            if s:
                break
            
        if not s:
            # TODO: Add a better error handling here.
            print('Failed to execute command.')
            
            
           
                
            