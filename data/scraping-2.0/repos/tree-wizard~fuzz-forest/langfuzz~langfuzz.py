import os
import openai
import re
import subprocess
import time
from typing import Tuple

from .langfuzzDB import Database
from .utils import run_atheris_fuzzer, num_tokens_from_string

class LangFuzz:
    def __init__(self, sqlitedb, language, base_prompt_path):
        self.db = Database(sqlitedb)
        self.sqlitedb = sqlitedb
        self.language = language
        self.base_prompt_path = base_prompt_path #replace with find_base_prompt_path()

    def find_base_prompt_path(self):
        if self.language == "python":
            return os.path.join('prompts', 'base-atheris-prompt.py')
        else:
            raise Exception(f"Language {self.language} not supported.")

    def generate_fuzz_code(self, prompt):
        max_retries = 5
        retry_delay = 5

        for _ in range(max_retries):
            try:
                response = openai.ChatCompletion.create(
                    model='gpt-4',
                    #model='gpt-3.5-turbo-16k',
                    messages=[
                        {"role": "system", "content": prompt}],
                    max_tokens=3500,
                    temperature=0.6,
                )
                return response["choices"][0]["message"]["content"]
            except openai.error.RateLimitError as e:
                print(f"Rate limit error encountered: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            except openai.error.APIConnectionError as e:
                print(f"Connection error encountered: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

        print("Failed to generate fuzz code after multiple retries.")
        return None

    def save_fuzz_test_to_db(self, library_name, function_name, fuzz_test_code):
        file_name = 'fuzz_' + function_name + '.py'
        tokens = num_tokens_from_string(fuzz_test_code)
        with self.db as db:
            db.save_fuzz_test_to_db(library_name, function_name, file_name, fuzz_test_code, tokens)

    def get_fuzz_tests_by_function_name(self, functions_list, runs=None, refactored=None, exception=None, instrumented=None):
        with self.db as db:
            return db.get_fuzz_tests_by_function_name(functions_list, runs, refactored, exception, instrumented)
    
    def update_fuzz_test_in_db(self, id, runs=None, run_output=None, coverage=None, exception=None, crash=None, contents=None, refactored=None, instrumented=None, deprecated=None):
        with self.db as db:
            db.update_fuzz_test_in_db(id, runs, run_output, coverage, exception, crash, contents, refactored, instrumented, deprecated)

    def get_existing_fuzz_file_data(self, library_name):
        with self.db as db:
            return db.get_existing_fuzz_file_data(library_name)
        
    # possibly refactor to use get_fuzz_tests_by_function_name
    def get_generated_functions_that_contain_string_in_contents(self, search_string):
        with self.db as db:
            return db.get_generated_functions_that_contain_string_in_contents(search_string)
    
    def get_functions_that_contain_string(self, library_name, search_string):
        with self.db as db:
            result = db.get_functions_that_contain_string(library_name, search_string)
        # Filter out None values and extract function names from the tuples
        function_names = [func[0] for func in result if func[0] is not None]
        return function_names
    
    def get_functions_by_filename(self, library_name, filename, fuzz_test=True):
        with self.db as db:
            result = db.get_functions_in_filename(library_name, filename, fuzz_test)
        # Filter out None values and extract function objects from the list
        functions = [func for func in result if func is not None]
        return functions
    
    def get_lib_fuzz_tests_from_db(self, library_name, runs=None, exception=None, refactored=None, instrumented=None):
        with self.db as db:
            return db.get_lib_fuzz_tests_from_db(library_name, runs, exception, refactored, instrumented)

    def get_radon_functions_from_db(self, lib_name, score=None):
        with self.db as db:
            return db.get_radon_functions_from_db(lib_name, score)

    def check_instrumentation(self):
        functions = self.get_generated_functions_that_contain_string_in_contents("atheris.instrument_all()")
        for function in functions:
            print(function.function_name)
            #mark as instrumented
            self.update_fuzz_test_in_db(function.id, instrumented=True)

    def create_prompt(self, base_template, fuzzer_context, library_name, function_name, function_code):
        if self.language == 'python':
            directive = f"Return ONLY valid and properly formatted Python code. Do NOT include any comments, explanations or notes. Import the {library_name} and write an atheris fuzz test for the {function_name} function in {library_name}:\n"
        else:
            directive = "Write a fuzz test for the following function:\n"

        exception_check = "Review function source and make sure to catch and ignore any EXPECTED exceptions that may arise from passing invalid input to the tested function. Avoid the blanket 'except Exception' and make them specific to the function being tested, use source code as reference" 

        prompt = base_template
        if fuzzer_context is not None:
            prompt += fuzzer_context
        prompt += f"{directive}This is the source code for{function_name}:{function_code}{exception_check}"

        num_tokens = num_tokens_from_string(prompt)

        if num_tokens < 4500:
            return prompt
        else:
            prompt = base_template + directive + exception_check
        
        num_tokens = num_tokens_from_string(prompt)

        if num_tokens <= 4500:
            return prompt

    def fix_code(self, code: str, output: str) -> str:
        retry_delay = 10
    
        while True:
            try:
                prompt = (
                    f"Please rewrite the following code to fix any issues or errors:\n\n"
                    f"---Code Starts---\n"
                    f"{code}\n"
                    f"---Code Ends---\n\n"
                    f"Output: {output}\n\n"
                    f"""IMPORTANT: Return only valid and properly formatted Python code. Do NOT include any comments, explanations or notes on the changes made."
                    f"IMPORTANT: Make sure the fuzzer always includes atheris.instrument_all() and executes it first like so
                    def main():
                    atheris.Setup(sys.argv, TestOneInput)
                    atheris.Fuzz()
                    if __name__ == \"__main__\":
                    atheris.instrument_all() # This is needed for coverage to work.
                    main()"""
                )
                response = openai.ChatCompletion.create(
                    model='gpt-4',
                    #model='gpt-3.5-turbo-16k',
                    messages=[
                        {"role": "system", "content": prompt}],
                    max_tokens=4500,
                    temperature=0.6,
                )
                return response["choices"][0]["message"]["content"]
            except openai.error.RateLimitError as e:
                print(f"Rate limit error encountered: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            except openai.error.APIConnectionError as e:
                print(f"Connection error encountered: {e}. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)

    def fix_fuzz_test(self, function_code, output) -> Tuple[str, str, bool]:
        successful_run = 'Done 2 runs'
        max_attempts = 5
        updated_code = function_code

        if num_tokens_from_string(function_code + output) < 4500:
            for attempt in range(max_attempts):
                print(f'fixing code, attempt {attempt}')
                updated_code = self.fix_code(updated_code, output)
                new_output = run_atheris_fuzzer(updated_code)
                #print(updated_code)

                if successful_run in new_output:
                    print('fixed')
                    return updated_code, new_output, True
                else:
                    output = new_output
                    time.sleep(1)
        print('Reached max_attempts without sucessfull run')
        return updated_code, output, False

    def fix_fuzz_test_code(self, library_name):
        fuzz_functions = self.get_lib_fuzz_tests_from_db(library_name, runs=False)
        for function in fuzz_functions:
            print("Fixing fuzz test code for", function.function_name)
            updated_code, output, runs = self.fix_fuzz_test(function.contents, function.run_output)
            #function to fix fuzz test code, returns fixed_code, run status, and output
            if runs == True:
                self.update_fuzz_test_in_db(function.id, runs=True, run_output=str(output), contents=updated_code, refactored=True)
            else:
                self.update_fuzz_test_in_db(function.id, runs=False, run_output=str(output), contents=updated_code, refactored=True) 

    def run_fuzzer(self, command, timeout):
        try:
            output = subprocess.check_output(command, shell=True, stderr=subprocess.STDOUT, text=True, timeout=timeout)
            crash = False
        except subprocess.CalledProcessError as e:
            output = e.output
            crash = True
        except subprocess.TimeoutExpired as e:
            output = e.output.decode()
            output += "Fuzzer Timeout"
            crash = False

        return output, crash

    def parse_coverage(self, output):
        cov_pattern = re.compile(r'cov:\s+(\d+)')
        cov_matches = cov_pattern.findall(output)
        if cov_matches:
            cov = int(cov_matches[-1])
        else:
            cov = None

        return cov

    def fuzz_functions_list(self, function_list, time=20000):
        generated_files_path = os.path.join('saved_repos', 'generated_files', 'fuzz')
        os.makedirs(generated_files_path, exist_ok=True)
        original_dir = os.getcwd()

        # get fuzz functions from db
        for function in function_list:
            function_path = os.path.join(generated_files_path, function.function_name)
            os.makedirs(function_path, exist_ok=True)
            fuzzer_file_path = os.path.join(function_path, function.file_name)

            with open(fuzzer_file_path, 'w') as fuzzer_file:
                fuzzer_file.write(function.contents)

            try:
                os.chdir(function_path)
                command = f'python {function.file_name} -max_total_time={time}'
                timeout = time + 100  # add 2 minute timeout to catch hangs

                output, crash = self.run_fuzzer(command, timeout)
                exception = 'exception' in output.lower()
                cov = self.parse_coverage(output)
                self.update_fuzz_test_in_db(function.id, run_output=output, coverage=cov, exception=exception, crash=crash)
            except Exception as e:
                print(f"Error running fuzzer for {function.function_name}: {e}")
            finally:
                os.chdir(original_dir)


    def extended_fuzz_analysis_by_filenames(self, functions_list, time=20000, refactored=None, exception=None, instrumented=None):
        generated_files_path = os.path.join('saved_repos', 'generated_files', 'fuzz')
        os.makedirs(generated_files_path, exist_ok=True)
        original_dir = os.getcwd()

        fuzz_functions = self.get_fuzz_tests_by_function_name(functions_list, runs=True, refactored=refactored, exception=exception, instrumented=instrumented)
        for function in fuzz_functions:
            function_path = os.path.join(generated_files_path, function.library_name, function.function_name)
            os.makedirs(function_path, exist_ok=True)
            print(function.function_name)
            #print(function.contents)
            fuzzer_file_path = os.path.join(function_path, function.file_name)

            with open(fuzzer_file_path, 'w') as fuzzer_file:
                fuzzer_file.write(function.contents)

            try:
                os.chdir(function_path)
                command = f'python {function.file_name} -max_total_time={time}'
                timeout = time + 100  # add 2 minute timeout to catch hangs

                output, crash = self.run_fuzzer(command, timeout)
                exception = 'exception' in output.lower()
                cov = self.parse_coverage(output)
                self.update_fuzz_test_in_db(function.id, run_output=output, coverage=cov, exception=exception, crash=crash)
            except Exception as e:
                print(f"Error running fuzzer for {function.function_name}: {e}")
            finally:
                os.chdir(original_dir)

    def extended_fuzz_analysis(self, library_name, time=20000, refactored=None, exception=None, instrumented=None):
        generated_files_path = os.path.join('saved_repos', 'generated_files', 'fuzz')
        os.makedirs(generated_files_path, exist_ok=True)
        original_dir = os.getcwd()
        fuzz_functions = self.get_lib_fuzz_tests_from_db(library_name, runs=True, refactored=refactored, exception=exception, instrumented=instrumented)

        for function in fuzz_functions:
            fuzz_test_path = os.path.join(generated_files_path, library_name, function.function_name)
            os.makedirs(fuzz_test_path, exist_ok=True)
            
            fuzzer_file_path = os.path.join(fuzz_test_path, function.file_name)
            print(fuzzer_file_path)
            with open(fuzzer_file_path, 'w') as fuzzer_file:
                fuzzer_file.write(function.contents)

            try:
                os.chdir(fuzz_test_path)
                command = f'python {function.file_name} -max_total_time={time}'
                timeout = time + 100  # add 2 minute timeout to catch hangs

                output, crash = self.run_fuzzer(command, timeout)
                exception = 'exception' in output.lower()
                cov = self.parse_coverage(output)
                self.update_fuzz_test_in_db(function.id, run_output=output, coverage=cov, exception=exception, crash=crash)
            except Exception as e:                                                              
                print(f"Error running fuzzer for {function.function_name}: {e}")
            finally:
                os.chdir(original_dir)


    def initial_fuzz_analysis(self, library):
        fuzz_functions = self.get_lib_fuzz_tests_from_db(library_name=library, runs=False, exception=False)
        print("Initial Fuzzing of " + str(len(fuzz_functions)) + " function for " + library)
        for function in fuzz_functions:
            print(function.function_name)
            output = run_atheris_fuzzer(function.contents)

            if 'Done 2 runs' in output:
                self.update_fuzz_test_in_db(function.id, runs=True, run_output=output)
            elif 'Python exception' in output:
                self.update_fuzz_test_in_db(function.id, runs=False, run_output=output, exception=True)
            elif 'deprecated' in output:
               self.update_fuzz_test_in_db(function.id, runs=False, run_output=output, deprecated=True)
            else:
                self.update_fuzz_test_in_db(function.id, runs=False, run_output=output)

    def add_fuzz_files_to_prompt(self, file_data):
        fuzz_prompt_context = ''
        for function in file_data:
            fuzz_prompt_context += f"Valid example fuzzers:\n{function.contents}\n"
        return fuzz_prompt_context
    
    def generate_fuzz_tests(self, library_name, function_list=None):
        base_template = open(self.base_prompt_path, "r").read()
        fuzz_file_data = self.get_existing_fuzz_file_data(library_name)
        fuzzer_context = self.add_fuzz_files_to_prompt(fuzz_file_data)
        with self.db as db:
            functions = db.get_functions_from_db(library_name, function_list)
                # Above was setup, below we create the full prompt, generate the fuzz test, and save it to the db
        for function in functions:
            complete_prompt = self.create_prompt(base_template, fuzzer_context, library_name, function.function_name, function.contents)
            if not self.db.generated_file_exists(library_name, function.function_name):
                print(function.function_name)
                fuzz_test_code = self.generate_fuzz_code(complete_prompt)
                if fuzz_test_code is not None:
                    self.save_fuzz_test_to_db(library_name, function.function_name, fuzz_test_code)

if __name__ == "__main__":
    http_libs = {
        'urllib3': {
            'github': 'https://github.com/urllib3/urllib3',
            'docs': 'https://urllib3.readthedocs.io/en/stable/'
    } }

    # Initialize LangFuzz object
    langfuzz = LangFuzz(sqlitedb, 'python', base_prompts_path)

    # This defines the score of radon complexity to pull from the database
    # A and B are the highest scores, E and F are the lowest scores
    radon_score = ['C', 'D', 'E', 'F']
    # radon_score is optional, if you don't pass it, it will pull all the functions
    priority_funcs = LangFuzz.get_radon_functions_from_db('urllib3', radon_score)
    langfuzz.generate_fuzz_tests('urllib3', priority_funcs)
