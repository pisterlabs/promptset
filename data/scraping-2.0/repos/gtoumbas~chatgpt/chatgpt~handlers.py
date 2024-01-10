import json
import importlib

class FunctionHandler:
    # TODO: Be smarter about context size
    def __init__(self, api_key, files_with_functions, example_path, model="gpt-3.5-turbo-0613", overwrite=False):
        self.api_key = api_key
        self.model = model
        self.functions = self.get_all_functions(files_with_functions)
        self.overwrite = overwrite

        # Read example file
        with open(example_path, "r") as f:
            example = f.read()
        self.example = example

        # Check if functions.json exists
        try:
            with open("functions.json", "r") as f:
                functions = json.load(f)
            
            if overwrite:
                self.create_function_json()
        except:
            self.create_function_json()


    def get_functions_from_file(self, file_path):
        # Get all text from file
        with open(file_path, "r") as f:
            text = f.read()
        
        # Look for def statementss and get all lines that are idented relative to it
        functions = []
        for line in text.split("\n"):
            if line.startswith("def"):
                functions.append(line)
            elif line.startswith("    "): # Will not work for nested functions
                functions[-1] += "\n" + line

        return functions
    
    def get_all_functions(self, files_with_functions):
        functions = {} # {File: [functions]}
        for file_path in files_with_functions:
            functions[file_path] = self.get_functions_from_file(file_path)
        return functions
    

    def create_function_json(self, model):
        from .chat_session import ChatSession
        print("Creating functions.json file...")
        chat = ChatSession(api_key=self.api_key, model=model, system_message=self.example)

        message = ""
        for file_path, functions in self.functions.items():
            message += f"FILENAME: {file_path}\n"
            for function in functions:
                message += function + "\n"

        
        json_functions = chat.send_message(message)
        # Make sure its a valid json
        try:
            test = json.loads(json_functions)
        except:
            raise Exception("The response from OpenAI was not a valid JSON. Please try again.")
        
        # Save to file
        with open("functions.json", "w") as f:
            f.write(json_functions)

        return json_functions
    
    def create_function_list_and_refs(self):
        # Creates from json file
        # CHeck if file exists
        function_refs = {}
        functions = []
        try:
            with open("functions.json", "r") as f:
                json_functions = json.load(f)
        except:
            raise Exception("No functions.json file found. Please run create_function_json() first.")

        for file_name in json_functions:
            module = importlib.import_module(file_name.replace(".py", ""))
            for function in json_functions[file_name]:
                name = function
                description = json_functions[file_name][function]["description"]
                parameters = json_functions[file_name][function]["parameters"]
                functions.append({
                    "name": name,
                    "description": description,
                    "parameters": parameters
                })
                function_refs[name] = getattr(module, name)

        return functions, function_refs


    