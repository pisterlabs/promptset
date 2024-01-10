import re
import subprocess
import time
import logging
from langchain.chat_models import ChatOpenAI
import os
import logging
import time
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import g4f

g4f.debug.logging = True  # Enable debug logging
g4f.debug.check_version = False  # Disable automatic version checking

load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
verbose_logging = os.getenv('VERBOSE', 'False') == 'True'

logging.basicConfig(level=logging.DEBUG, filename='run.log', filemode='w')
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')

script_filename = ""

class FolderNameGenerator:
    def __init__(self):
        self.llm = OpenAI(temperature=0.9, openai_api_key=openai_api_key)
        self.prompt = PromptTemplate(
            input_variables=["query"],
            template="The Query is ' {query} ', Provide me a title in the format --<Title here>--. Only the folder name and not path or anything else. Note: Give me a 10-15 character folder name."
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)

    def generate_unique_foldername(self, query):
        logging.info("Generating unique folder name...")
        folder_name = self.chain.run({"query": query})
        logging.debug(f"LLM Response for folder name: {folder_name}")
    
        print("foldername", folder_name)

        match = re.search(r'--(.*?)--', folder_name)
        print("match", match)

        if match:
            folder_name = match.group(1).strip()
            timestamp = str(int(time.time()))
            return f"{folder_name}{timestamp}"
        else:
            timestamp = str(int(time.time()))
            return f"{query.replace(' ', '_')}{timestamp}"

class OtherFilesGenerator:
    def __init__(self, api_key, is_offline=False):
        self.is_offline = is_offline
        if not is_offline:
            self.llm = ChatOpenAI(openai_api_key=api_key)

        
    def predict(self, prompt):
        if self.is_offline:
            # Using g4f for predictions when offline
            response = g4f.ChatCompletion.create(
                model=g4f.models.gpt_4,
                messages=[{"role": "user", "content": prompt}],
                provider=g4f.Provider.Liaobots,
            )
            # Directly return the response string
            return response
        else:
            return self.llm.predict(prompt)
        
    def generate_other_files(self, code):
        logging.info("Checking if other files are required... for running the code")
        data =  self.predict(code+"' this is the code i wrote but its not running becoz it needs some other files, what other files are required to run this code? please provide the Code, FileName and Extension of the file. Provide the code")
        # print("data", data)
        return data
    
class DynamicAgent:
    
    def __init__(self, api_key, is_offline=False):
        self.is_offline = is_offline
        if not is_offline:
            self.llm = ChatOpenAI(openai_api_key=api_key)
        logging.info("DynamicAgent initialized.")

    def predict(self, prompt):
        if self.is_offline:
            # Using g4f for predictions when offline
            response = g4f.ChatCompletion.create(
                model=g4f.models.gpt_4,
                messages=[{"role": "user", "content": prompt}],
                provider=g4f.Provider.Liaobots,
            )
            # Directly return the response string
            return response
        else:
            return self.llm.predict(prompt)


    def generate_unique_foldername(self, query):
       logging.info("Generating unique folder name...")
       folder_name_generator = FolderNameGenerator()
       return folder_name_generator.generate_unique_foldername(query)
    
    def generate_unique_filename(self):
        # use LLM to generate a unique filename by providing the q
        timestamp = str(int(time.time()))
        return f"temp_script_{timestamp}.py"

    def generate_other_files(self, code):
        # use the OtherFilesGenerator  to generate other files
        other_files_generator = OtherFilesGenerator(None, is_offline=self.is_offline)
        return other_files_generator.generate_other_files(code)
    
    def create_files_from_string(self,input_string, folder_path):
        code_blocks = re.findall(r'```([a-zA-Z]+)\n(.*?)```', input_string, re.DOTALL)
        file_name_pattern = r'<title>(.*?)<\/title>'
        try:
            for extension, code in code_blocks:
                # Extract file name if available
                file_name_match = re.search(file_name_pattern, code)
                if file_name_match:
                    file_name = file_name_match.group(1)
                else:
                    file_name = "unnamed"
                
                # Create folder if it doesn't exist
                full_folder_path = os.path.join(folder_path, extension)
                if not os.path.exists(full_folder_path):
                    os.makedirs(full_folder_path)
                
                # Create file and write code to it
                with open(f"{full_folder_path}/{file_name}.{extension}", 'w') as f:
                    f.write(code)
        except Exception as e:
            print("Error in create_files_from_string", e)
            return "Error in create_files_from_string"
    
    def determine_task(self, query):
        logging.info("Determining task from the provided query...")
        # Replace self.llm.predict with self.predict
        return self.predict(f"So tell me How can I achieve this in Python {query}, Provide me only one complete code nothing else with if you need output to you?")


    def extract_executable_code(self, full_code):
        logging.info("Extracting code segment...")
        extracted_code = re.search(r'```python(.*?)```', full_code, re.DOTALL)
        
        if extracted_code:
            logging.debug(f"Extracted code: {extracted_code.group(1).strip()}")
            return extracted_code.group(1).strip()
        return ""

    def validate_extracted_code(self, code):
        logging.info("Validating the extracted code...")
        logging.debug(f"Validating code: {code}")
        return bool(self.extract_executable_code(code))

    def fetch_and_validate_solution(self, query):
        logging.info("Fetching and validating solution...")
        full_solution = self.determine_task(query)
        logging.debug(f"Full solution: {full_solution}")
        if self.validate_extracted_code(full_solution):
            return full_solution
        else:
            return self.extract_executable_code(full_solution)

    def detect_dependencies(self, code):
        logging.info("Detecting dependencies required for the generated code...")
        detected_imports_response = self.predict(f"What are the pip installables for this code, provide as copy-paste snippet? {code}")
        dependencies = re.findall(r'pip install ([\w\-]+)', detected_imports_response)
        logging.debug(f"Detected dependencies: {dependencies}")
        return dependencies

    def ensure_dependencies_installed(self, dependencies):
        logging.info("Ensuring all detected dependencies are installed...")
        for package in dependencies:
            try:
                logging.debug(f"Installing {package}...")
                subprocess.check_output(['pip', 'install', package])
            except subprocess.CalledProcessError:
                logging.error(f"Failed to install {package}")
                return f"Failed to install {package}"
        return "Dependencies installed successfully."

    def execute_code(self, code, other_files):
        logging.info("Executing the generated code...")
        executable_code = self.extract_executable_code(code)
        logging.debug(f"Executable code: {executable_code}")
        
        if not executable_code:
            return "Error: Code extraction failed.", None, None
    
        # Display the code to the user for review and ask for confirmation
        print("Generated code to execute:\n")
        print(executable_code)
        execute_confirmation = input("Do you want to execute this code? (y/N): ")
        if execute_confirmation.lower() != 'y':
            return "Execution aborted by the user.", None, None
    
        script_filename = self.generate_unique_filename()
    
        with open(script_filename, 'w') as f:
            f.write(executable_code)
    
        # Create other files
        self.create_files_from_string(other_files, ".")
    
        try:
            output = subprocess.check_output(['python', script_filename])
            return output.decode('utf-8').strip(), None, script_filename
        except subprocess.CalledProcessError as e:
            error_message = e.output.decode('utf-8').strip()
            return f"Error during execution: {error_message}", executable_code, script_filename
    

    def fix_and_execute_code(self, error, code):
        for _ in range(5):
            logging.info("Attempting to fix the code...")
            fix_instruction = self.predict(f"fix this error: '{error}' in the following code: ```{code}``` and Provide the correct complete code")
            dependencies = self.detect_dependencies(fix_instruction)
            self.ensure_dependencies_installed(dependencies)
            fixed_code = self.extract_executable_code(fix_instruction)

            if not fixed_code:
                continue

            # folder_name = self.generate_unique_foldername()
            # os.makedirs(folder_name, exist_ok=True)  # ensure the folder exists
            script_filename = self.generate_unique_filename("temp")

            with open(script_filename, 'w') as f:
                f.write(fixed_code)

            try:
                output = subprocess.check_output(['python', script_filename])
                return output.decode('utf-8').strip()
            except subprocess.CalledProcessError as e:
                error = e.output.decode('utf-8').strip()
                with open(os.path.join("temp", "final_attempted_code.py"), 'w') as f:
                    f.write(fixed_code)
            return f"Error during re-execution after fixing: {error}"
        
    def prompt_delete_files(self, folder_path):
       response = input("Would you like to delete the created files? (y/N): ")
       if response.lower() == 'y':
        #    delete the file using
           os.remove(folder_path)
           print("Files deleted.")
       else:
           print("Files kept.")
        
    def process_query(self, query):
        logging.info("Processing the user query...")
        code_to_run = self.fetch_and_validate_solution(query)
        other_files = self.generate_other_files(code_to_run)
        detected_dependencies = self.detect_dependencies(code_to_run)
        dependencies_status = self.ensure_dependencies_installed(detected_dependencies)
        if "failed" in dependencies_status.lower():
            return dependencies_status
        result, failed_code, script_filename = self.execute_code(code_to_run, other_files)
        
        if failed_code:
            result = self.fix_and_execute_code(result, failed_code)
        self.prompt_delete_files(script_filename)
        
        return result