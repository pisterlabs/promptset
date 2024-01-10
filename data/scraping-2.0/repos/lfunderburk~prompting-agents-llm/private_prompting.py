import openai

class Prompter:
    def __init__(self, api_key, gpt_model, temperature=0.2):
        if not api_key:
            raise Exception("Please provide the OpenAI API key")

        self.api_key  = api_key
        self.gpt_model = gpt_model
        self.temperature = temperature
    
    def chat_completion(self, messages: list):
        openai.api_key = self.api_key
        response = openai.ChatCompletion.create(model=self.gpt_model, 
                                                messages=messages,
                                                temperature=self.temperature)
        return response["choices"][0]["message"]["content"]
    
    def natural_language_with_roles(self, db_name:str, schema:str, natural_question:str):

        system_content = f"You are a data analyst, and you specialize in solving business questions with SQL.\
                        You are given a natural language question, and your role is to translate the question\
                        into a query that can be executed against a database. \
                        Ensure your queries are written in a single line, with no special characters"
        user_content = f"Please generate a SQL query for data with in a database named {db_name}\
                        along with a schema {schema} for the question {natural_question}"

        full_prompts = [
                                {"role" : "system", "content" : system_content},
                                {"role" : "user", "content" : user_content},
                                ]
        
        result = self.chat_completion(full_prompts)

        return result
   
    
    def natural_language_zero_shot(self, db_name:str, schema:str, natural_question:str):
        user_content= f"Answer the question {natural_question} for table {db_name} with schema {schema}"
        
        full_prompt = [{"role" : "assistant", "content" : user_content}]
        
        
        result = self.chat_completion(full_prompt)

        return result
    
    def natural_language_single_shot(self, db_name:str, schema:str, natural_question:str):
        prompt= f"Answer the question {natural_question} for table {db_name} with schema {schema}\
                       Question: How many records are there?\
                       Answer: SELECT COUNT(*) FROM bank"
        
        full_prompt = [{"role" : "assistant", "content" : prompt}]
        
        
        result = self.chat_completion(full_prompt)

        return result
    
    def natural_language_few_shot(self, db_name:str, schema:str, natural_question:str):
        prompt= f"Answer the question {natural_question} for table {db_name} with schema {schema}\
                Question: How many records are there?\
                Answer: SELECT COUNT(*) FROM bank\
                Question: Find all employees that are unemployed\
                Answer: SELECT * FROM bank WHERE job = 'unemployed'"
        
        full_prompt = [{"role" : "assistant", "content" : prompt}]
        
        
        result = self.chat_completion(full_prompt)

        return result
    