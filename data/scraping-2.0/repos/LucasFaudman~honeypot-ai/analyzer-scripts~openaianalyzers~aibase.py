

from analyzerbase import *


import ast
from openai import OpenAI, OpenAIError
import tiktoken

from .tools import TOOLS
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')


class OpenAIAnalyzerBase:
    
    def __init__(self, training_data_dir=Path("openai-training-data"), aidb_path=Path("tests/aidb"), api_key=OPENAI_API_KEY, model="gpt-4-1106-preview") -> None:
        

        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = model
        
        self.training_data_dir = Path(training_data_dir)
        if not self.training_data_dir.exists():
            self.training_data_dir.mkdir(exist_ok=True, parents=True)
        
        self.aidb_path = Path(aidb_path)
        if not self.aidb_path.exists():
            self.aidb_path.mkdir(exist_ok=True, parents=True)






    def _try_openai(self, getter_fn, extractor_fn=lambda x: x, **kwargs):
        """
        Tries to get a response from OpenAI and extract the result. Catches 
        OpenAIError and other exceptions if they occur when extracting the desired result.
        """

        try:
            response = getter_fn(**kwargs)
        except OpenAIError as e:
            print(f'OpenAI Error: {e}')
            return f'OpenAI Error: {e}', e

        try:
            result = extractor_fn(response)
        except Exception as e:
            print(f'Error extracting result: {e}')
            return response, f'Error extracting result: {e}'
        

        return response, result



    def _try_load_json_result(self, result):
        """
        Tries to load a result as json. If that fails, tries to load it as a python literal.
        If that fails, returns a dict with the errors and the result.
        """

        try:
            return json.loads(result)
        except Exception as e1:
            try:
                return ast.literal_eval(result)
            except Exception as e2:
                return {"error_json_loads": e1, 
                        "error_ast_literal_eval": e2, 
                        "result": result}



    def num_tokens_from_messages(self, messages, disallowed_special=()):
        """Returns the number of tokens used by a list of messages."""
        try:
            encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
              encoding = tiktoken.get_encoding("cl100k_base")
        if self.model:
            num_tokens = 0
            for message in messages:
                num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
                for key, value in message.items():
                    num_tokens += len(encoding.encode(value, disallowed_special=disallowed_special))
                    if key == "name":  # if there's a name, the role is omitted
                        num_tokens += -1  # role is always required and always 1 token
            
            num_tokens += 2  # every reply is primed with <im_start>assistant
            return num_tokens, (num_tokens / 1000) * .01
        else:
            raise NotImplementedError(f"""num_tokens_from_messages() is not presently implemented for model {self.model}.
    See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
        


    def recursively_make_serializeable(self, obj):
        """Recursively makes an object serializeable by converting it to a dict or list of dicts and converting all non-string values to strings."""
        serializeable_types = (str, int, float, bool, type(None))
        if isinstance(obj, dict):
            return {k: self.recursively_make_serializeable(v) for k,v in obj.items()}
        elif isinstance(obj, list):
            return [self.recursively_make_serializeable(item) for item in obj]
        elif not isinstance(obj, serializeable_types):
            return str(obj)
        else:
            return obj


    def format_content(self, content):
        """Formats content for use a message content. If content is not a string, it is converted to json."""
        if not isinstance(content, str):
            content = self.recursively_make_serializeable(content)
            content = json.dumps(content, indent=0)

        return content
            


    def index_content(self, content):
        """Indexes items in sequence into dict with indices as keys."""
        return dict(enumerate(content))
        



    def write_training_data(self, filename, data):
        """Utility used to write hardcode training data to a file."""

        file = (self.training_data_dir / filename)
        if not file.parent.exists():
            file.parent.mkdir(exist_ok=True, parents=True)
        
        if isinstance(data, list):
            data = "\n".join(data)
        
        if isinstance(data, dict):
            if all(isinstance(k, str) and isinstance(v, str) for k,v in data.items()):
                data = "\n".join([f"{k}\n{v}\n" for k,v in data.items()])
            else:
                data = json.dumps(data, indent=4)
            
        with file.open("w+") as f:
            f.write(data)

    
       
    def read_training_data(self, filename, returnas=None):
        """
        Reads training data from a file and returns it as a list, dict, json 
        or split_firstline which is (firstline, data) 
        or just read as string if returnas is None or str.
        """

        file = self.training_data_dir / filename

        with file.open("r") as f:
            
            if returnas in ("list", list):
                return [line.rstrip("\n") for line in f]
            
            elif returnas in ("dict", dict):
                return {k:v for k,v in zip(f.readlines()[::2], f.readlines()[1::2])}
            
            elif returnas in ("json", json):
                return json.load(f)
            
            elif returnas in ("split_firstline", "mw"):
                lines = f.readlines()
                firstline = lines[0].rstrip("\n")
                data = "".join(lines[1:])

                return firstline, data
        
            else:
                return f.read()