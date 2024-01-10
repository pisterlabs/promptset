import openai
import pandas as pd

class PlannerAgent:
    def __init__(self, function_name_descriptions: list[tuple], api_key: str):
        """
        Class initialization function.

        Parameters:
        function_name_descriptions (list): A list of functions that can be used during planning
        api_key (str): OpenAI API key for making completion API calls.
        """
        self.function_name_descriptions = function_name_descriptions
        self.api_key = api_key

    def __call__(self, prompt: str) -> str:
        """
        Takes in user query and returns an OpenAIObject which has the steps to take to achieve the goal in the user query.

        Parameters:
        prompt (str): User query text to be used for planning 

        Returns:
        openai.openai_object.OpenAIObject: A OpenAIObject object from OpenAI API call
        """
        completion = openai.chat.completions.create(
            model="gpt-4-1106-preview",
            # max_tokens=100,
        messages=[
            {
            "role": "system",
            "content": f'''
            You are a planner agent. You will be provided with a question by the user that is trying to run a python simulation. 
            Your job is to provide the steps to answer the user query using the python functions available to you. If the functions available cant solve the problem then you should tell the user that you cant solve the problem.
            Your methodology is: Reason about how many steps it will take to get to the answer (20 words or less). Then give the output in the following format: ``` [use the function_name1 to do task 1, use function_name2 to do task 2] ``` 
            Always remember to use the triple backticks and the square brackets for the final answer!
            
            These are the functions available to you: {self.function_name_descriptions}.

            Here are some examples of user queries and the steps you should take to answer them:
            user: whats the current value of xyz?
            assistant: Since the user query is asking for the current value of a param, I can get that using 1 step using the model_info function. ``` [use the function model_info to fetch the xyz parameter] ```
            user: What is the current value of all params?
            assistant: Since the user query is asking for the current value of all params I can get that using 1 step using the model_info function. ``` [use the function model_info to fetch all the parameters] ```
            user: What are the assumptions in this model?
            assistant: Since the user query is asking for contextual information about the model it must be in the model_documentation and I can get this in one step. ``` [use the function model_documentation to fetch the metrics and params in the model.] ```
            user: What are the columns in the dataframe?
            assistant: Since the user query is asking to analyze the column in the current dataframe I should use the analyze_dataframe function in one step.  ``` [use the function analyze_dataframe to fetch the columns in the dataframe.] ```
            user: What would happen to the A column at the end of the simulation if my xyz param was 20?
            assistant: Since the user query has a multistep question here which involves changing a param and then analyzing the dataframe column I should plan 2 steps for this. ``` [we use function change_param to change the xyz parameter to 20, use function analyze_dataframe to get the A at the end of the simulation.] ```
            user: What is the current value of my xyz param? can you change it to 50 and tell me what the A column at the end of the simulation would be?
            assistant: Since the user query has a multistep question which involves fetching the current value, changing a param and analyzing the dataframe I should take 3 steps. ``` [use function model_info to fetch the crash_chance parameter, use function change_param to change the xyz parameter to 50, use function analyze_dataframe to get the A at the end of the simulation.] ```
            '''
            },
            {
            "role": "user",
            "content": prompt
            },
        ],
        )
        reply = completion.choices[0].message.content

        return reply



class ExecutorAgent:
    def __init__(self, df: pd.DataFrame, function_schemas: list, params: dict, api_key: str):
        """
        Class initialization function.

        Parameters:
        df (pd.DataFrame): A DataFrame object that represents some data used for completion
        function_schemas (list): A list of functions that can be used during completion
        api_key (str): Key for API calls, necessary for function completion.
        """
        self.df = df
        self.function_schemas = function_schemas
        self.api_key = api_key
        self.params = params
        self.messages = [
                {"role": "system", 
                 "content": f'''You are an executor agent that helps with python function calling. The parameters of the model are {self.params}'''},
                ],
        

    def __call__(self, prompt: str) -> str:
        """
        Converts the plan to a OpenAIObject which will have function calling instructions.

        Parameters:
        prompt (str): Text to be used for completion.

        Returns:
        openai.openai_object.OpenAIObject: A OpenAIObject object from OpenAI API call
        """
        
        completion = openai.chat.completions.create(
            model="gpt-3.5-turbo-0613",
            messages=self.messages[0],
                
            # add function calling
            functions=self.function_schemas,
            function_call="auto",  # specify the function call
        )
        reply= completion.choices[0].message
        return reply
    
    def add_message(self, message: dict):
        """
        Adds a message to the messages list.

        Parameters:
        message (str): Message to be added to the messages list.
        """
        self.messages[0].append(message)

    def delete_all_messages(self):
        """
        Deletes all messages from the messages list.
        """
        self.messages = [
                {"role": "system", 
                 "content": f'''You are an executor agent that helps with python function calling. Think of which function to call and the arguments to send based on user query AND the previous messages . The parameters of the model are {self.params}'''},
                ],