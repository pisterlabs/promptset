import openai
import dotenv
import os
from prompts import EXECUTOR_PROMPT
import traceback
from retry import retry
import re

dotenv.load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")
openai.organization = os.environ.get("OPENAI_ORG")

overloading_code = """
printed_statements = []
def print(*args):
    printed_statements.append(str(args))
"""

def extract_python_code_blocks(text):
    pattern = r'```python(.*?)```'
    matches = re.findall(pattern, text, re.DOTALL)
    return [match.strip() for match in matches]

class Executor():
    def __init__(self, data_description, data_location, data_headers, subset):
        self.data_location = data_location
        self.data_description = data_description
        self.data_headers = data_headers
        self.subset = subset

    def generate_code(self, steps):
        messages = [{"role": "system", "content": EXECUTOR_PROMPT(self.data_description, self.data_location, self.data_headers, self.subset)}]
        messages.append({"role": "user", "content": steps})
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages
            )
        code = response["choices"][0]["message"]["content"]
        return code
    
    def fix_code(self, steps, code, error):
        messages = [{"role": "system", "content": EXECUTOR_PROMPT(self.data_description, self.data_location, self.data_headers, self.subset)}]
        messages.append({"role": "user", "content": steps})
        messages.append({"role": "assistant", "content": code})
        messages.append({"role": "user", "content": f"The code throws the following error. Please fix it an rewrite the entire code. Do not paraphrase and only provide the code starting with ```python and ending with ```.\n\nError: {error}"})
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages
            )
        code = response["choices"][0]["message"]["content"]
        return code
    
    def execute(self, code):
        env = {}
        exec(code, env)
        return env
        
    def process_instructions(self, steps, max_fixes=5, max_attempts=3):
        for _ in range(max_attempts):
            code = self.generate_code(steps)
            for _ in range(max_fixes):
                try: 
                    print("Executing code")
                    env = self.execute(overloading_code + extract_python_code_blocks(code)[0])
                    return env["printed_statements"]
                except Exception as e:
                    print("Fixing code")
                    code = self.fix_code(steps, code, f"{e}, {traceback.print_exc()}")
            print("Regenerating code")
        raise ValueError("Code generation and fixing attempts exceeded")


steps = """
step 1: Load the dataset into a suitable format, e.g., a pandas DataFrame
step 2: Handle any missing data (e.g., replace '?' by NaN, drop rows with missing data, or impute missing values)
step 3: Calculate the frequency distribution of the different work classes and education levels
step 4: Calculate the cross-tabulation (contingency table) between work class and education level [report]
step 5: Calculate the Pearson's Chi-squared test statistic and p-value to determine the dependence between work class and education level [report]
step 6: Calculate the Cramér's V coefficient to measure the strength of the association between work class and education level [report]
"""

code = '''```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency

# Step 1: Load the dataset into a pandas DataFrame
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
column_names = ["Age", "Work Class", "fnlwgt", "Education", "Education Num", "Marital Status",
                "Occupation", "Relationship", "Race", "Sex", "Capital Gain", "Capital Loss",
                "Hours per Week", "Native Country", "Income"]
df = pd.read_csv(url, header=None, names=column_names, sep=',\s', na_values=["?"], engine='python')

# Step 2: Handle missing data
df = df.dropna()

# Step 3: Calculate the frequency distribution of work classes and education levels
work_class_freq = df['Work Class'].value_counts()
education_freq = df['Education'].value_counts()

# Step 4: Calculate the cross-tabulation between work class and education level
crosstab = pd.crosstab(df['Work Class'], df['Education'])

# Report results
print("Cross-tabulation:", crosstab)

# Step 5: Calculate the Pearson's Chi-squared test statistic and p-value
chi_stat, p_value, _, _ = chi2_contingency(crosstab)

# Report results
print("Chi-squared test statistic: ", chi_stat)
print("p-value: ", p_value)

# Step 6: Calculate the Cramér's V coefficient
n = crosstab.sum().sum()
min_dim = min(crosstab.shape) - 1
cramers_v = np.sqrt(chi_stat / (n * min_dim))

# Report results
print("Cramér's V coefficient: ", cramers_v)
```'''

subsample = """
39, State-gov, 77516, Bachelors, 13, Never-married, Adm-clerical, Not-in-family, White, Male, 2174, 0, 40, United-States, <=50K
50, Self-emp-not-inc, 83311, Bachelors, 13, Married-civ-spouse, Exec-managerial, Husband, White, Male, 0, 0, 13, United-States, <=50K
38, Private, 215646, HS-grad, 9, Divorced, Handlers-cleaners, Not-in-family, White, Male, 0, 0, 40, United-States, <=50K
"""

if __name__ == "__main__":
    data_description = "This is a dataset from the 1994 consensus"
    data_location = "data/adult.data.csv"
    headers = ["age","workclass","fnlwgt","education","education-num", "marital-status","occupation", "relationship","race","sex", "capital-gain","capital-loss","hours-per-week","native-country","makes"]
    executor =  Executor(data_description, data_location, headers, subsample)
    print(executor.process_instructions(steps))
    # # print(executor.generate_code(steps))
    # # print(code)
    # code = executor.generate_code(steps)
    # print(code)
    # env = executor.execute(code)
    # print(env["printed_statements"])
    # import pdb; pdb.set_trace()
 