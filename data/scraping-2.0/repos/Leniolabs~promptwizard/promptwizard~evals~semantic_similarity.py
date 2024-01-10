from ..openai_calls import openai_call
from ..cost import input, output, embeddings
import concurrent.futures
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import threading
from typing import List, Dict

class semanticSimilarity:
    def __init__(self, test_cases: List[Dict], prompts: List[str], model_test: str="gpt-3.5-turbo", model_test_temperature: float=0.6, model_test_max_tokens: int=1000, model_embedding: str="text-embedding-ada-002", best_prompts: int=2, timeout: int=10, n_retries: int=5):

        """
        Initialize a semantic_similarity instance.

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

        Specify in the prompts you generate that the response must be text.

        Most importantly, output NOTHING but the prompt. Do not include anything else in your message."""
        self.prompts = prompts
        self.model_embedding = model_embedding
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
    
    def calculate_embedding(self, test_case):
        response_embedding = openai_call.create_embedding(self.model_embedding, test_case['output'], self.timeout, self.n_retries)
        test_case['embedding'] = response_embedding.data[0].embedding
        tokens_embedding = response_embedding.usage.total_tokens
        return tokens_embedding
    
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
                - Tokens used for embeddings.
        """

        cost = 0
        tokens_input = 0
        tokens_output = 0
        tokens_embedding = 0
        prompt_results = {prompt: {'score': 0} for prompt in self.prompts}
        results = [{"method": "Semantic Similarity"}]

        if 'embedding' not in self.test_cases[0]:
            threads = []

            results_embedding = []

            for test_case in self.test_cases:
                thread = threading.Thread(target=lambda t=test_case: results_embedding.append(self.calculate_embedding(t)))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

            tokens_embedding = sum(results_embedding)


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
                    embedding_response = openai_call.create_embedding(self.model_embedding, result_content, self.timeout, self.n_retries)
                    embedding_test_case = test_case['embedding']
                    similarity_score = cosine_similarity(np.array(embedding_response.data[0].embedding).reshape(1, -1) , np.array(embedding_test_case).reshape(1, -1))[0, 0]
                    
                    prompt_results[prompt]['score'] = prompt_results[prompt]['score'] + similarity_score
                    tokens_embedding = tokens_embedding + embedding_response.usage.total_tokens

                    prompt_and_results.append({"test": test_case['input'], "answer": result_content, "ideal": ideal_output, "result": similarity_score})
                
                results.append(prompt_and_results)
                prompt_and_results = []

        cost_input = input.cost(tokens_input, self.model_test)
        cost_output = output.cost(tokens_output, self.model_test)
        cost_embeddings = embeddings.cost(tokens_embedding, self.model_embedding)
        cost = cost_input + cost_output + cost_embeddings

        # Sort the prompts by score.
        data_list = []
        for i, prompt in enumerate(self.prompts):
            score = prompt_results[prompt]['score']/len(self.test_cases)
            data_list.append({"prompt": prompt, "rating": score})

        sorted_data = sorted(data_list, key=lambda x: x['rating'], reverse=True)
        best_prompts = sorted_data[:self.best_prompts]
        sorted_data.append(results)
        return sorted_data, best_prompts, cost, tokens_input, tokens_output, tokens_embedding
    
    def evaluate_optimal_prompt(self):

        """
        Evaluate the optimal prompt by testing candidate prompts and selecting the best ones.

        Returns:
            tuple: A tuple containing the result data, best prompts, cost, input tokens used, output tokens and embedding tokens used.
        """
        
        return self.test_candidate_prompts()