from sklearn.metrics import roc_curve, RocCurveDisplay
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import warnings
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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

    def __init__(self, df, max_tokens=4000, temperature=0.5):
        self.open_ai_api_key = None
        self.df = df
        self.column_descriptions = None
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.conversation_history = []

    def count_gpt_prompt_tokens(self, messages, model="gpt-3.5-turbo"):
        '''Returns the number of tokens used by a list of messages.'''
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        if model == "gpt-3.5-turbo":
            return self.count_gpt_prompt_tokens(messages, model="gpt-3.5-turbo-0301")
        elif model == "gpt-3.5-turbo-0301":
            tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1  # if there's a name, the role is omitted        
        elif model == "gpt-4":
            return self.count_gpt_prompt_tokens(messages, model="gpt-4-0314")
        elif model == "gpt-4-0314":
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
        num_tokens = 0
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
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
            # load from provided key
            self.open_ai_api_key = key
        else:
            print("ERROR: no key or filename provided")
            return
    
    def set_column_descriptions(self, descriptions):
        '''Set the descriptions for the columns in the dataframe'''
        self.column_descriptions = descriptions

    def get_python_code_from_openai(self, instruction, the_model = "gpt-3.5-turbo", verbose_level = 0):
        '''Send an instruction to the OpenAI API and return the Python code that is returned'''
        # build system prompt
        system_prompt =  "You are a Python programming assistant\n"
        system_prompt += "The user provides an instruction and you return Python code to achieve the desired result\n"    
        system_prompt += "You document the code by adding comments\n"
        system_prompt += "You include all necessary imports\n"
        system_prompt += "It is very important that you start each piece of code in your response with [code] and end it with [/code]\n"
            
        # build user prompt
        if self.column_descriptions is not None:
            user_prompt = "I have a dataframe df with the following columns:\n"
            for col in self.column_descriptions:
                user_prompt += f"# {col}: {self.column_descriptions[col]}\n"                
            user_prompt += "Produce Python code for the following instruction(s):\n"
            user_prompt += instruction

        # build the message array for the request
        msg_array = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
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

        # use regular expression to extract the pieces of code; each piece starts with [code] and ends with [/code]
        code = re.findall(r'\[code\](.*?)\[\/code\]', content, re.DOTALL)
        code = "\n".join(code)

        # remove any lines that try to read data from a file
        code = "\n".join([line for line in code.split("\n") if not line.startswith("df = pd.read_csv")])

        # replace lines that start with "#" with print() statements
        # code = "\n".join([('print("' + line.replace("#", "") + '")') if line.startswith("#") else line for line in code.split("\n")])

        # add "import pandas as pd" if it is not already there
        if not "import pandas" in code:
            code = "import pandas as pd\n" + code

        # print content and nr of tokens in prompt as counted and as reported by OpenAI
        if verbose_level > 1:
            print(f"Content: {content}")
            print(f"Number of tokens in prompt: {self.count_gpt_prompt_tokens(msg_array, model=the_model)}")
            print(f"Number of tokens reported by OpenAI: {response['usage']['prompt_tokens']}")

        return code

    def execute_code(self, code):
        # Make the DataFrame accessible to the code
        namespace = {'df': self.df}

        # Try executing the code
        try:
            exec(code, namespace)
        except Exception as e:
            # Handle execution errors
            return f"Error executing code: {str(e)}"

        # After successful execution, extract the 'result' variable from the namespace
        result = namespace.get('result', None)

        # Save the result summary in the conversation history
        self.conversation_history[-1]['result_summary'] = str(result)

        return result

    def free_text_instruction(self, instruction):
        '''Execute a free text instruction'''
        print("Waiting for response from openai API...")
        code = self.get_python_code_from_openai(instruction)
        print("Returned code:")
        print("-----------------")
        print(code)
        print("-----------------")
        print("Executing the code...\n")
        try:
            exec(code, {'df': self.df})
        except Exception as e:
            print(f"Error when executing the code: {e}")

    def generate_analysis_code(self, instruction):
        '''Generate code for analyzing the data'''
        print("Generating code for instruction: " + instruction)
        code, content, token_cnt = self.get_python_code_from_openai(instruction)
        print(f"Code:\n{code}\n")

    def perform_binary_classification_analysis(self, target_column, model_names = ['LogisticRegression'], metric = 'accuracy', cv_folds = 5, n_iter = 20, verbose_level = 1, show_warnings = False, show_plots = True):
        '''Perform classification analysis'''
        if not show_warnings:
            warnings.filterwarnings('ignore')
        # first check that the target column is binary
        if len(self.df[target_column].unique()) != 2:
            print(f"ERROR: target column {target_column} is not binary")
            return
        # next, check that the models in model_names are valid
        valid_models = ["LogisticRegression", "RandomForest", "GradientBoosting", "KNeighbors", "SVC"]
        for model_name in model_names:
            if model_name not in valid_models:
                print(f"ERROR: model {model_name} is not valid")
                return
        # next, check that the metric is valid
        valid_metrics = ["accuracy", "precision", "recall", "f1"]
        if metric not in valid_metrics:
            print(f"ERROR: metric {metric} is not valid")
            return
        # split into X and y
        X = self.df.drop([target_column], axis=1)
        y = self.df[target_column]
        # split into train and test
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        # specify column transformer (leave binary columns untouched, scale continuous features, one-hot-encode non-numerical features)
        binary_cols = [col for col in X_train.columns if X_train[col].nunique() == 2]
        continuous_cols = [col for col in X_train.columns if col not in binary_cols]
        non_numerical_cols = [col for col in X_train.columns if X_train[col].dtype == 'object']
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), continuous_cols),
                ('passthrough', 'passthrough', binary_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), non_numerical_cols)
            ])
        # define the model and parameter spaces
        for model_name in model_names:
            if model_name == 'LogisticRegression':
                model = LogisticRegression()
                param_grid = {'classifier__C': [0.1, 1, 10, 100, 1000], 'classifier__penalty': ['l1', 'l2']}
            elif model_name == 'RandomForest':
                model = RandomForestClassifier()
                param_grid = {'classifier__n_estimators': [100, 200, 300, 400, 500], 'classifier__max_features': ['auto', 'sqrt', 'log2']}
            elif model_name == 'GradientBoosting':
                model = GradientBoostingClassifier()
                param_grid = {'classifier__n_estimators': [100, 200, 300, 400, 500], 'classifier__learning_rate': [0.001, 0.01, 0.1, 1, 10]}
            elif model_name == 'KNeighbors':
                model = KNeighborsClassifier()
                param_grid = {'classifier__n_neighbors': [3, 5, 7, 9, 11], 'classifier__weights': ['uniform', 'distance']}
            elif model_name == 'SVC':                
                model = SVC()
                param_grid = {'classifier__C': [0.1, 1, 10, 100, 1000], 'classifier__kernel': ['linear', 'rbf']}
            # create a pipeline
            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
            # perform randomized search
            if verbose_level > 0:
                print(f"Performing randomized search for {model_name}...")
            search = RandomizedSearchCV(pipeline, param_grid, cv=cv_folds, scoring=metric, n_iter=n_iter, verbose=verbose_level)
            search.fit(X_train, y_train)
            # print the results
            if verbose_level > 0:
                print(f"Best parameter (CV score={search.best_score_:.3f}):")
                print(search.best_params_)
                # print the test score
                print(f"Test score: {search.score(X_test, y_test):.3f}")
            # print the classification report
            y_pred = search.predict(X_test)
            if verbose_level > 0:
                print(classification_report(y_test, y_pred))
                # print the confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                print(f"Confusion Matrix: \n{cm}")
            if verbose_level > 0:
                print("\n") 
            # plot ROC curve
            if show_plots:
                try:
                    fig, ax = plt.subplots(figsize=(6, 6))
                    RocCurveDisplay.from_estimator(search, X_test, y_test, ax=ax)
                    ax.plot([0, 1], [0, 1], linestyle='-', lw=2, color='k', alpha=.8)
                    ax.set_title(f"ROC curve for {model_name}")
                    ax.set_xlabel("False Positive Rate")
                    ax.set_ylabel("True Positive Rate")
                    ax.legend(loc="lower right")                    
                    plt.show()
                except:
                    print("ERROR: could not plot ROC curve")
        # reset warnings
        warnings.filterwarnings('default')
    
    