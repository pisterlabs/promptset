from dotenv import load_dotenv
import os
import openai
import subprocess
import pandas as pd

df = pd.read_csv('extracted-data.csv')

data_string = df.to_string()

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

chat_completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", 
    messages=[
        {"role": "system", "content": "You are a python expert in data analysis"},
        {"role": "user", "content": 
        f'''Pretend you are a python data analsyis,

            I want you ot give only the python code as a output so for example:

            Example 1:
            Input: How to print something
            Output: print("hello world")

            Example 2:
            Input: How to create a bar chart
            Output:
            import matplotlib.pyplot as plt
            categories = ['A', 'B', 'C']
            values = [10, 15, 7]
            plt.bar(categories, values)
            plt.show()

            Example 3:
            Input: how to craete a scatter plot in pyton
            Output:
            import matplotlib.pyplot as plt
            x_values = [1, 2, 3, 4, 5]
            y_values = [5, 7, 6, 8, 7]
            plt.scatter(x_values, y_values)
            plt.xlabel('X Values')
            plt.ylabel('Y Values')
            plt.title('Scatter Plot')
            plt.show()

            Example 4:
            Input: Create me a bar chart example using plotpy
            Output:
            import plotly.graph_objects as go
            fruits = ['Apples', 'Oranges', 'Bananas', 'Grapes', 'Berries']
            quantities = [10, 15, 7, 10, 5]
            fig = go.Figure([go.Bar(x=fruits, y=quantities)])
            fig.update_layout(title_text='Fruit Quantities', xaxis_title='Fruit', yaxis_title='Quantity')
            fig.show()



            -----

            Visualise this data using plotly as a most appropriate visualization: {data_string}

'''}
        ]
    )

execute = chat_completion['choices'][0]['message']['content']

with open('python-execute.txt', 'w') as f:
    f.write(execute)

subprocess.run(["python", 'python-execute.txt'])