import pandas as pd
import openai
import tiktoken
import re

class AIDataAnalyzer:
    
    ''' 
    AIDA: Artificial Intelligence Data Analyzer 

    This class contains methods that uses the OpenAI API to analyze data.

    The idea is as follows:
    - The data are stored in a dataframe, which is passed to the class
    - Thereafter, the user can call methods to analyze the data
    - The analysis is done by consulting the OpenAI API for generating code, which is then executed
    - The results are returned to the user

    The class has several pre-specified methods for analyzing data, but the user can also give free text instructions

    In essence, the class automates the process of typing a question "give me code for the following ..." into ChatGPT 
    and then copy-pasting the returned code into a Jupyter notebook
    '''

    def __init__(self, df, max_tokens=4000, temperature=0.2, api_key=None, api_key_filename=None, column_descriptions=None):
        self.open_ai_api_key = self.set_openai_api_key(key=api_key, filename=api_key_filename)
        self.df = df
        self.column_descriptions = column_descriptions
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.comments = []
        self.query_history = []
        self.code_history = []
        self.current_query = ""
        # display a reminder to set openai key in case not set
        print("Note: Beware that there may be security risks involved in executing AI-generated code. Use the 'query' function at own risk!")
        if self.open_ai_api_key is None:
            print("Note: No OpenAI API key has been set yet. Use set_openai_api_key() to set it.")
        if self.column_descriptions is None:
            print("Note: No column descriptions have been set, which may negatively affect the results. Use set_column_descriptions() to set them.")

    def count_gpt_prompt_tokens(self, messages, model="gpt-3.5-turbo"):
        '''Returns the number of tokens used by a list of messages.'''
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
            
        tokens_per_message = 3
        tokens_per_name = 1
        if model in ["gpt-3.5-turbo", "gpt-3.5-turbo-0301"]:
            tokens_per_message = 4
            tokens_per_name = -1
        
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with assistant
        return num_tokens

    def set_openai_api_key(self, key=None, filename=None):
        '''Set the OpenAI API key. If a filename is provided, load the key from that file.'''
        # if a filename is provided, load the key from that file
        if filename is not None:
            # load from specified file
            try:
                with open(filename, "r") as f:
                    self.open_ai_api_key = f.read()
            except FileNotFoundError:
                print(f"ERROR: could not find {filename}")     
        elif key is not None:
            self.open_ai_api_key = key
        else:
            self.open_ai_api_key = None
        if self.open_ai_api_key is not None:
            # show a masked version of the open ai key (first 4 and last 4 characters, with . in between)
            print(f"OpenAI API key was set to {self.open_ai_api_key[0:4]}....{self.open_ai_api_key[-4:]}")
            
    def set_column_descriptions(self, descriptions):
        '''Set the descriptions for the columns in the dataframe'''
        for column in descriptions:
            if column not in self.df.columns:
                raise ValueError(f"The column '{column}' does not exist in the DataFrame.")
        self.column_descriptions = descriptions

    def get_python_code_from_openai(self, instruction, the_model = "gpt-3.5-turbo", verbose_level = 0, include_comments = True):
        '''Send an instruction to the OpenAI API and return the Python code that is returned'''
        # build system prompt
        system_prompt =  "You are a Python programming assistant\n"
        system_prompt += "The user provides an instruction and you return Python code to achieve the desired result\n"        
        if include_comments:
            system_prompt += "You document the code by adding comments\n"
        else:
            system_prompt += "You don't add comments to the code\n"
        system_prompt += "You include all necessary imports\n"
        system_prompt += "When displaying results in text form, you round to 3 decimals, unless specified otherwise\n"
        system_prompt += "When creating plots, you always label the axes and include a legend, unless specified otherwise\n"
        #system_prompt += "When creating plots in a loop, you organize them in 1 figure with 3 plots per row, unless specified otherwise\n"
        system_prompt += "It is very important that you start each piece of code in your response with [code] and end it with [/code]\n"
        system_prompt += "Assume that the dataframe uses named labels for columns, not integer indices.\n"
        system_prompt += "You can make use of IPython Markdown to beautify the outputs\n"
        system_prompt += "Here is an example of how to make a correlation heatmap: mask = np.triu(np.ones_like(correlation_matrix, dtype=bool)), heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', mask=mask, ax=axs[0], annot_kws={'size': 18})\n"

        # build user prompt
        user_prompt = ""
        user_prompt += "I would like your help with data analysis.\n"
        user_prompt += "The data consists of the following columns:\n"
        for col in self.column_descriptions:
            user_prompt += f"# {col}: {self.column_descriptions[col]}\n"
        if len(self.comments) > 0:
            user_prompt += "Here are some comments about the data and your task:\n"
            for comment in self.comments:
                user_prompt += f"# {comment}\n"
        user_prompt += "Please complete the following code:\n\n"
        user_prompt += "import pandas as pd\n"
        user_prompt += "df = pd.read_csv('data.csv')\n"
        user_prompt += "# " + instruction.replace("\n", "\n# ") + "\n\n"
        #user_prompt += "Remember to organize plots in 1 figure with 3 plots per row when creating them in a loop\n"
        user_prompt += "Remember that you must start each piece of code in your response with [code] and end it with [/code]\n"
        user_prompt += "Remember to include all necessary imports\n"
        user_prompt += "When you loop over df.columns, use cnt and col_idx as variable names\n"
        user_prompt += "Never use 'col_idx' as a variable name for something else\n"
        user_prompt += "When you compute the row or column for axs, call the variables subplot_row and subplot_col\n"
        user_prompt += "Remember to remove unused subplots\n"


        # build the message array for the request
        msg_array = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},            
        ]

        # count number of tokens in input
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
        encoding.encode(user_prompt)
        
        # perform the request
        openai.api_key = self.open_ai_api_key
        response = openai.ChatCompletion.create(
            model=the_model,
            messages=msg_array,
            max_tokens=self.max_tokens - self.count_gpt_prompt_tokens(msg_array, model=the_model),
            temperature=self.temperature,
        )
        
        # process the response
        token_cnt = response['usage']['total_tokens']
        content = response['choices'][0]['message']['content']
        code = re.findall(r'\[code\](.*?)\[\/code\]', content, re.DOTALL)
        code = "\n".join(code)

        # remove any lines that in which read_csv is called
        code = "\n".join([line for line in code.split("\n") if "read_csv" not in line])
        code = "\n".join([line for line in code.split("\n") if "# read the data" not in line])

        # add "import pandas as pd" if it is not already there
        if not "import pandas" in code:
            code = "import pandas as pd\n" + code
        # make sure that display and MarkDown are imported if Markdown appears anywhere in the code:
        if "Markdown" in code:
            code = "from IPython.display import display, Markdown\n" + code

        # print content and nr of tokens in prompt as counted and as reported by OpenAI
        if verbose_level > 1:
            print(f"Total tokens: {token_cnt}")
            print(f"Content: {content}")
            print(f"Number of tokens in prompt: {self.count_gpt_prompt_tokens(msg_array, model=the_model)}")
            print(f"Number of tokens reported by OpenAI: {response['usage']['prompt_tokens']}")

        return code

    def build_query(self, new_line):
        '''Add a line to the current query'''
        self.current_query += new_line + "\n"

    def reset_query(self):
        '''Reset the current query'''
        self.current_query = ""
    
    def show_query(self):
        '''Print the current query'''
        print(self.current_query)

    def execute_query(self, run_code = True, verbose_level = 0):
        '''Execute a free text instruction'''
        self.query_history.append(self.current_query)
        print("Query was submitted. Waiting for response...")
        code = self.get_python_code_from_openai(self.current_query, verbose_level = verbose_level, include_comments = (not run_code) or (verbose_level > 0))
        print("\nReturned code:")
        print("--------------------------------------------------------------------------")
        print(code.strip("\n"))
        print("--------------------------------------------------------------------------")
        self.code_history.append(code)
        if run_code:
            print("\nExecuting the code...\n")
            try:
                namespace = {'df': self.df}
                exec(code, namespace)
            except Exception as e:
                print(f"Error executing code: {str(e)}")
        self.current_query = ""

    #------------ comments -----------#
    def add_comment(self, comment):
        '''Add a comment to the list of comments unless it is already in there'''
        if comment not in self.comments:
            self.comments.append(comment)
    
    def show_comments(self):
        '''Print all comments'''
        cnt=0
        for comment in self.comments:
            print(f"[{cnt}] {comment}")
            cnt += 1

    def delete_comment(self, comment_nr):
        '''Delete a comment'''
        if comment_nr == -1:
            self.comments = []
        if comment_nr < len(self.comments):
            del self.comments[comment_nr]

    #------------ query history -----------#
    def show_query_history(self):
        '''Print the query history'''
        cnt=0
        for query in self.query_history:
            print(f"[{cnt}] {query}")
            cnt += 1

    def show_code_history(self):
        '''Print the code history'''
        cnt=0
        for code in self.code_history:
            print(f"\n[{cnt}]\n{code}")
            cnt += 1
