import openai
import dotenv
import os
import sys
from executor import Executor
from load_data import get_prompt_data

dotenv.load_dotenv()
openai.api_key = os.environ.get("OPENAI_API_KEY")
openai.organization = os.environ.get("OPENAI_ORG")

from prompts import EXAMPLE_DATA, CONTROLLER_SYSTEM_PROMPT

class Controller:

    def __init__(self) -> None:
        self.state = [
            {"role": "system", "content": CONTROLLER_SYSTEM_PROMPT},
        ]

    def load_data(self) -> None:
        # Adds information about the dataset to the state
        self.data_description = EXAMPLE_DATA
        print("Loading data")
        data = get_prompt_data('data/adult.names', 'data/adult.data.csv')
        data_location = "data/adult.data.csv"
        self.executor =  Executor(data["summary"], data_location, data["headers"], data["subset"][:3])

    def plan(self, user_request) -> None:
        # Generate a plan for testing the data
        print("Generating plan")
        self.state.append({"role": "user", "content": f"You are provided the following data:\n\n{self.data_description}\n\nYour client requests the following insight(s):\n{user_request}"})
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=self.state
            )
        plan = response["choices"][0]["message"]["content"]
        return plan
    
    def execute_plan(self, plan):
        print("Executing plan")
        results = self.executor.process_instructions(plan)
        # results = ["('Frequency distribution of workclass categories:\\n',  Private             22696\n Self-emp-not-inc     2541\n Local-gov            2093\n ?                    1836\n State-gov            1298\n Self-emp-inc         1116\n Federal-gov           960\n Without-pay            14\n Never-worked            7\nworkclass                1\nName: workclass, dtype: int64, '\\n')", "('Frequency distribution of education categories:\\n',  HS-grad         10501\n Some-college     7291\n Bachelors        5355\n Masters          1723\n Assoc-voc        1382\n 11th             1175\n Assoc-acdm       1067\n 10th              933\n 7th-8th           646\n Prof-school       576\n 9th               514\n 12th              433\n Doctorate         413\n 5th-6th           333\n 1st-4th           168\n Preschool          51\neducation            1\nName: education, dtype: int64, '\\n')", "('Mean education levels by workclass categories:\\n', workclass\n ?                  NaN\n Federal-gov        NaN\n Local-gov          NaN\n Never-worked       NaN\n Private            NaN\n Self-emp-inc       NaN\n Self-emp-not-inc   NaN\n State-gov          NaN\n Without-pay        NaN\nworkclass           NaN\nName: education_numeric, dtype: float64, '\\n')", "('Trends observed in the summary table and plots:\\n',)", '("1. The workclass \'Without-pay\' has the highest mean education level, followed by \'Self-employed-inc\' and \'Federal-gov\'.",)', "('2. Private sector workers have middle range mean education levels.',)", '("3. The workclass \'Never-worked\' has the lowest mean education level.",)', '("4. In the box plot, work classes \'Self-employed-inc\', \'Federal-gov\', and \'Local-gov\' have higher median education levels.",)', '("5. The range of education levels is widest for \'Self-employed-inc\' and \'Federal-gov\' work classes, showing a diverse workforce.",)']
        return results 
    
    def reflect(self) -> None:
        pass

if __name__ == "__main__":

    # user_request = "What is the relationship between work class and eduction?"
    user_request = sys.argv[1]
    print(f"User request: {user_request}")

    controller = Controller()
    controller.load_data()
    plan = controller.plan(user_request)
    print(f"Plan:\n{plan}")

    result = controller.execute_plan(plan)
    # print(f"Result:\n{result}")

    messages = [{"role": "system", "content": f"Given a user request and a poorly formatted response from a data anlaytics engine provide a direct answer using the data to back it up. Provide as much specifc quanitative evidence in your explination but do not state anything that is not reinforced by the data you are given."}, {"role": "user", "content": f"User request: {user_request}\n\nData:\n{result}"}]
    while True:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages
        )
        answer = response["choices"][0]["message"]["content"]
        print(answer)
        messages.append({"role": "assistant", "content": answer})
        follow_up = input(">>> ")
        messages.append({"role": "user", "content": follow_up})
