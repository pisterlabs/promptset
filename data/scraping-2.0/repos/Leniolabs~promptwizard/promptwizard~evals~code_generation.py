from ..openai_calls import openai_call
from ..cost import input, output
import ast
import concurrent.futures
import builtins
import js2py
import re
import os
from typing import List, Dict

class codeGeneration:
    def __init__(self, test_cases: List[Dict], prompts: List[str], model_test: str='gpt-3.5-turbo', model_test_temperature: float=0.6, model_test_max_tokens: int=1000, best_prompts: int=2, timeout: int=10, n_retries: int=5):

        """
        Initialize a Code Generation instance.

        Args:
            test_cases (list): List of test cases to evaluate.
            number_of_prompts (int): Number of prompts to generate and/or test.
            model_test (str): The language model used for testing.
            model_test_temperature (float): The temperature parameter for the testing model.
            model_test_max_tokens (int): The maximum number of tokens allowed for the testing model.
            model_generation (str): The language model used for generating prompts.
            model_generation_temperature (float): The temperature parameter for the generation model.
            prompts (list): List of prompts to evaluate.
            best_prompts (int): Number of best prompts to consider.

        Note:
            The 'system_gen_system_prompt' attribute is predefined within the class constructor.
        """

        self.test_cases = test_cases
        self.model_test = model_test
        self.model_test_temperature = model_test_temperature
        self.model_test_max_tokens = model_test_max_tokens
        self.system_gen_system_prompt = """Your job is to generate system prompts for GPT, given a description of the use-case and some test cases.

        In your generated prompt, you should describe how the AI should behave in plain English. Include what it will see, and what it's allowed to output. Be creative in with prompts to get the best possible results. The AI knows it's an AI -- you don't need to tell it this.

        You will be graded based on the performance of your prompt... but don't cheat! You cannot include specifics about the test cases in your prompt. Any prompts with examples will be disqualified.

        Specify in your prompts that you are going to generate that your task is to be a code generator, so the result to the prompt you are going to generate can only be a script in some programming language and the arguments you are going to use in your created function.

        These are examples: "def count(l):\n return len(l)", "def suma_lista(lista):\n total = 0\n for elemento in lista:\n\n total += elemento\n return total", "def factorial(n):\n if n == 0:\n\n return 1\n else:\n\n return n * factorial(n - 1)", "def es_par(numero):\n return numero % 2 == 0", "def maximo_lista(lista):\n if len(lista) == 0:\n\n return None\n else:\n\n return max(lista)", That is the only valid output format for the prompt.", Note you don't need to use print() or console.log().

        Most importantly, output NOTHING but the prompt. Do not include anything else in your message."""
        self.prompts = prompts
        self.best_prompts = best_prompts
        self.timeout = timeout
        self.n_retries = n_retries

    def process_prompt(self, prompt, test_case, model, model_max_tokens, model_temperature):
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"{test_case['input']}"}
        ]
        response = openai_call.create_chat_completion(model, messages, model_max_tokens, model_temperature, 1, timeout=self.timeout, n_retries=self.n_retries)
        partial_tokens_input = response.usage.prompt_tokens
        partial_tokens_output = response.usage.completion_tokens
        result_content = response.choices[0].message.content
        
        return partial_tokens_input, partial_tokens_output, result_content, test_case['output']

    def test_candidate_prompts(self):

        """
        Test a list of candidate prompts with test cases and evaluate their performance.

        Returns:
            tuple: A tuple containing the following elements:
                - List of results and statistics.
                - List of best-performing prompts.
                - Cost of generating and testing prompts.
                - Tokens used for input.
                - Tokens used for output.
        """

        cost = 0
        tokens_input = 0
        tokens_output = 0
        prompt_results = {prompt: {'correct': 0, 'total': 0} for prompt in self.prompts}
        results = [{"method": "Code Generation"}]

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
        
            for prompt in self.prompts:
                prompt_and_results = [{"prompt": prompt}]
                for test_case in self.test_cases:
                    
                    future = executor.submit(
                        self.process_prompt,
                        prompt,
                        test_case,
                        self.model_test,
                        self.model_test_max_tokens,
                        self.model_test_temperature
                    )
                    futures.append(future)
                    partial_tokens_input, partial_tokens_output, result_content, ideal_output = future.result()
                    tokens_input += partial_tokens_input
                    tokens_output += partial_tokens_output  
                    
                    if 'python' in test_case['input'].lower():
                        try:
                        
                            
                            with open("functions.py", "w") as file:
                                parsed_code = ast.parse(result_content)
                                has_function = any(isinstance(stmt, ast.FunctionDef) for stmt in parsed_code.body)
                                if has_function:
                                    
                                    file.write(ast.unparse(parsed_code))
                            
                            
                            with open("functions.py", "r") as file:
                                function_code = file.read()
                        
                        
                            exec(function_code, builtins.__dict__)

                        
                            name_function = None
                            for stmt in reversed(parsed_code.body):
                                if isinstance(stmt, ast.FunctionDef):
                                    name_function = stmt.name
                                    break
                            if name_function is not None:

                                
                                executable_function = builtins.__dict__[name_function]

                            
                                result = executable_function(*ast.literal_eval(test_case['arguments'])) 
                            
                            
                            if result == ideal_output:
                                prompt_results[prompt]['correct'] += 1
                                prompt_and_results.append({"test": test_case['input'], "answer": result_content, "ideal": ideal_output, "result": True})
                                prompt_results[prompt]['total'] += 1
                            if result != ideal_output:
                                prompt_and_results.append({"test": test_case['input'], "answer": result_content, "ideal": ideal_output, "result": "Final result incorrect."})
                                prompt_results[prompt]['total'] += 1

                        except Exception as e:
    
                            prompt_results[prompt]['total'] += 1
                            prompt_and_results.append({"test": test_case['input'], "answer": result_content, "ideal": ideal_output, "result": "Not Python script."})
                    
                    if 'javascript' in test_case['input'].lower():
                        
                        try:
                            file = "functions.js"
                            with open("functions.js", "w") as file:
                                file.write(result_content)

                            with open("functions.js", "r") as f:
                                js_code = f.read()
                            
                            context = js2py.EvalJs()

                            context.execute(js_code)

                            functions = re.findall(r"function\s+(\w+)\s*\(", js_code)

                            for function_name in functions:
                                
                                js_function = context.eval(function_name)
                                result = js_function(*ast.literal_eval(test_case['arguments']))
                            
                                def convert_to_native(js_object, target_type):
                                
                                    conversion_functions = {
                                        int: lambda x: int(x),
                                        float: lambda x: float(x),
                                        str: lambda x: str(x),
                                        bool: lambda x: bool(x),
                                        list: lambda x: list(x),
                                        dict: lambda x: dict(x),
                                        tuple: lambda x: tuple(x),
                            
                                    }

                                
                                    if target_type in conversion_functions:
                                
                                        conversion_function = conversion_functions[target_type]
                                        result = conversion_function(js_object)
                                        return result
                                    
                                final_result = convert_to_native(result, type(ideal_output))

                            if final_result == ideal_output:
                                prompt_results[prompt]['correct'] += 1
                                prompt_and_results.append({"test": test_case['input'], "answer": result_content, "ideal": ideal_output, "result": True})
                                prompt_results[prompt]['total'] += 1
                            if final_result != ideal_output:
                                prompt_and_results.append({"test": test_case['input'], "answer": result_content, "ideal": ideal_output, "result": "Final result incorrect."})
                                prompt_results[prompt]['total'] += 1

                        except Exception as e:

                            prompt_results[prompt]['total'] += 1
                            prompt_and_results.append({"test": test_case['input'], "answer": result_content, "ideal": ideal_output, "result": "Not Javascript script."})

                results.append(prompt_and_results)
                prompt_and_results = []

        cost_input = input.cost(tokens_input, self.model_test)
        cost_output = output.cost(tokens_output, self.model_test)
        cost = cost + cost_input + cost_output

        # Calculate and print the percentage of correct answers and average time for each model
        best_prompt = self.prompts[0]
        best_percentage = 0
        data_list = []
        for i, prompt in enumerate(self.prompts):
            correct = prompt_results[prompt]['correct']
            total = prompt_results[prompt]['total']
            percentage = (correct / total) * 100
            data_list.append({"prompt": prompt, "rating": percentage})
            print(f"Prompt {i+1} got {percentage:.2f}% correct.")
            if percentage >= best_percentage:
                best_percentage = percentage
                best_prompt = prompt
        sorted_data = sorted(data_list, key=lambda x: x['rating'], reverse=True)
        best_prompts = sorted_data[:self.best_prompts]
        print(f"The best prompt was '{best_prompt}' with a correctness of {best_percentage:.2f}%.")
        sorted_data.append(results)
        return sorted_data, best_prompts, cost, tokens_input, tokens_output
    
    def evaluate_optimal_prompt(self):

        """
        Evaluate the optimal prompt by testing candidate prompts and selecting the best ones.

        Returns:
            tuple: A tuple containing the result data, best prompts, cost, input tokens used, and output tokens used.
        """

        result = self.test_candidate_prompts()
        file_to_delete_python = "functions.py"
        if os.path.isfile(file_to_delete_python):
            os.unlink(file_to_delete_python)
        file_to_delete_javascript = "functions.js"
        if os.path.isfile(file_to_delete_javascript):
            os.unlink(file_to_delete_javascript)

        return result