"""
This file aims to act as an Idea Evaluator for ideas that would boost the Circular Economy 
"""

import csv
import os
import csv 
from openai import OpenAI
import re
import random
import plotly.graph_objects as go 
import time

class IdeaEvaluator:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.OpenAI_key = self.load_openai_key()
        self.client = OpenAI(api_key=self.OpenAI_key)
        #populating dataset 
        self.rows = []
        self.populate_rows(self.rows)
        self.baseline_metrics  = {'Market Potential':20, 'Scalability':20,'Feasibility':20,'Maturity Stage':20,'Technological Innovation':20}

        #baseline model data 
        self.baseline_model_data = []
        self.categories = {}
        self.fieldnames=['Index', 'Problem', 'Solution', 'Market Potential', 'Scalability', 'Feasibility','Maturity Stage','Technological Innovation', 'Combined Score', 'Category']
        self.new_metrics = ['Market Potential', 'Scalability', 'Feasibility','Maturity Stage','Technological Innovation']
        self.user_weights = []
        self.user_model_data = []
        

    def populate_rows(self, rows):
        with open(self.dataset_path, encoding = 'latin-1') as file:
            csv_reader = csv.reader(file)
            header = next(csv_reader)
            for row in csv_reader:
                rows.append(row)
        
    def load_openai_key(self):
        try:
            with open(".env", "r") as env_file:
                lines = env_file.readlines()
                for line in lines:
                    key, value = line.strip().split("=")
                    if key.strip() == "OPENAI_KEY":
                        return value.strip()
        except FileNotFoundError:
            print(".env file not found.")
        except Exception as e:
            print(f"Error loading OPENAI_KEY: {e}")
        return None
    
    def generate_results(self, idx, problem, solution, metrics):
        score = int(100 / len(metrics))

        metricList = ''
        for metric in metrics:
            metricList = metricList + metric + ", "
        metricList = metricList[:-2]

        example = ''
        total_score = 0
        for metric in metrics:
            random_score = random.randint(1, score)
            total_score += random_score
            example = example + metric + ": " + str(random_score) + " "
        example = example + "Combined Score: " + str(total_score) + " Category: Construction" 
        messages = [
            {
                "role": "system",
                "content": '''You are an AI-powered decision-support tool used to evaluate innovative circular economy business opportunities.
                You are given a problem statement and a solution. Here are a few important metrics you need to evaluate these solutions on, 
                Metrics : ''' + metricList + '''. Follow these steps for the output :
                Step 1 : For each metric, you provide a score for the solution between 0 and ''' + str(score) + '''. The higher the score, the better the solution.
                Step 2 : You must create a combined score, by aggregating (sum of) all the individual scores from the metrics above. This score should be between 0 and 100.
                Step 3 : You are going to categorize the given problem into a category relevant to strengthening the circular economy. Only mention the category name, and not the description.
                Ensure each criteria is given equal weightage, and is scored out of ''' + str(score) + '''. Ensure that the output has scores for all of the ''' + str(len(metrics)) + ''' metrics. Ensure that the output is in one line always, do not add newline characters. Ensure that the output is exactly the same format 
                as the example, with the same number of spaces and punctuation. You do not have to show your reasoning for the scores.''',
            },
            {
                "role": "user",
                "content": '''Problem Statement : The construction industry is indubitably one of the significant contributors to global waste, contributing approximately 1.3 billion tons of waste annually, exerting significant pressure on our landfills and natural resources. Traditional construction methods entail single-use designs that require frequent demolitions, leading to resource depletion and wastage.
                            Solution : Herein, we propose an innovative approach to mitigate this problem: Modular Construction. This method embraces recycling and reuse, taking a significant stride towards a circular economy. Modular construction involves utilizing engineered components in a manufacturing facility that are later assembled on-site. These components are designed for easy disassembling, enabling them to be reused in diverse projects, thus significantly reducing waste and conserving resources. Not only does this method decrease construction waste by up to 90%, but it also decreases construction time by 30-50%, optimizing both environmental and financial efficiency. This reduction in time corresponds to substantial financial savings for businesses. Moreover, the modular approach allows greater flexibility, adapting to changing needs over time. We believe, by adopting modular construction, the industry can transit from a 'take, make and dispose' model to a more sustainable 'reduce, reuse, and recycle' model, driving the industry towards a more circular and sustainable future. The feasibility of this concept is already being proven in markets around the globe, indicating its potential for scalability and real-world application.''',
            },
            {
                "role": "assistant",
                "content": example,
            },
            {
                "role": "user",
                "content": "Problem Statement : " + problem + " Solution : " + solution,
            }
        ]
        
        res = self.client.chat.completions.create(
            model = "gpt-3.5-turbo",
            messages = messages
        )
        msg = res.choices[0].message.content
        
        tokens = msg.split(': ')

        result = [idx, problem, solution]
        
        for x in range(1, len(metrics) + 2): # offset by 2 because it starts at 1 and need extra token for combined score
                result.append(tokens[x].split()[0])
            
        result.append(tokens[-1])
        return result

    def baseline_model(self):
        for row in self.rows:
            baseline_row = self.generate_results(row[0], row[1], row[2], ['Market Potential', 'Scalability', 'Feasibility','Maturity Stage','Technological Innovation'])
            self.baseline_model_data.append(baseline_row)
        self.baseline_model_data.sort(key=lambda x: x[-2], reverse=True)


        with open('./data/baseline_results.csv','w', newline = '', encoding = 'latin-1') as file:
            writer = csv.writer(file, self.fieldnames)
            writer.writerow(self.fieldnames)
            writer.writerows(self.baseline_model_data)

    def populate_categories(self):
        for row in self.baseline_model_data:
            category = row[-1]
            if category in self.categories:
                self.categories[category] += 1 
            else:
                self.categories[category] = 1

    def filter_categories(self, model, category):
        filter = []
        for row in model:
            if category == row[-1]:
                filter.append(row)
        fields = ['Index', 'Problem', 'Solution']
        fields.extend(self.new_metrics)
        fields.extend(['Combined Score', 'Category'])
        with open(f"./data/filtered_{category}_results.csv","w" ,newline = '', encoding = 'latin-1') as file:
            writer = csv.writer(file, fields)
            writer.writerow(fields)
            writer.writerows(filter)

    def bar_visualization(self):
        keys = list(self.categories.keys())
        values = list(self.categories.values())

        fig = go.Figure(data=[go.Bar(x=keys, y=values)])
        fig.update_layout(title_text='Category Distribution', xaxis_title='Categories', yaxis_title='Frequency')
        fig.show()

    def user_model(self):
        while True:
            print("Tell us about yourself, explain what type of investments you are looking for")
            print('''e.g. Im a young investor looking to make big profit, I have a large amount of money to invest and am willing to try anything for a big profit margin and need a return within the next 10 years\n''')
            intro = input()
        
            chat_completion = self.client.chat.completions.create(
                messages=[
                    { "role": "system", "content": "You are a decision-support tool, given an investor profile determine weightings ideas as an integer from 1 to 100 based on how relevant each of the following metrics is: Market Potential, Scalibility, Feasibility, Maturity Stage, Technological Innovation. " },
                    { "role": "user", "content": "I am a Venture Capital Analyst looking for start-ups, I am looking for safe investments and I would need my investment to pay off in 3-5 years."},
                    { "role": "assistant", "content": "23, 90, 63, 74, 9" },
                    { "role": "user", "content": intro}
                ],
                model="gpt-3.5-turbo",
            )
            weights = chat_completion.choices[0].message.content
            #weights = '70, 85, 50, 67, 94' 

            if(weights[0].isdigit()): # check for error/other prompt response
                break

        values = list(map(lambda x: int(x), weights.split(', ')))

        print("Tell us more about the type of ideas you want to invest in?")
        print("i.e. are you interested in a certain sector (Education), businesss model etc. \n")
        goals = input()

        chat_completion = self.client.chat.completions.create(
            messages=[
                { "role": "system", "content": "You are a decision-support tool, given an investor profile and goals determine 5 new metrics that would be relevant to the investors interests and rate their importance as an integer from 1 to 100. Ensure the weightings exactly match the example punctuation and format and all output is on one line." },
                { "role": "user", "content": "I am a Venture Capital Analyst looking for start-ups, I am looking for safe investments and I would need my investment to pay off in 3-5 years. I want to focus on midsize companies and products that consist of energy based environmental solutions"},
                { "role": "assistant", "content": "Regulatory Compliance,90,Sustainibility Impact,50,Company Partnerships,73,Customer Retention,61,Government Incentives,43" },
                { "role": "user", "content": intro + " " + goals}
            ],
            model="gpt-3.5-turbo",
        )
        tokens = chat_completion.choices[0].message.content.split(",")

        print("\nCreating metrics that are specific to you...\n")
        for x in range(0, 9, 2):
            self.new_metrics.append(tokens[x])
        
        for x in range(1, 10, 2):
            values.append(int(tokens[x]))
            
        return values
    
    def evaluateAdditionalMetrics(self):
        fieldnames=['Index', 'Problem', 'Solution']
        fieldnames.extend(self.new_metrics)
        fieldnames.append('Combined Score')
        
        for row in self.rows:
            baseline_row = self.generate_results(row[0], row[1], row[2], self.new_metrics)
            self.user_model_data.append(baseline_row)
        self.user_model_data.sort(key=lambda x: x[::-1], reverse=True)

        print("The New Metrics are: ", self.new_metrics[5:])

    def calculateScore(self):
        average = sum(self.user_weights) / len(self.user_weights)

        weightedScore = [[] for x in range(len(self.user_model_data))]

        for x in range(0, len(self.user_model_data)):
            total = 0
            
            for y in range(0, 10):
                weightedScore[x].append(int(float(self.user_model_data[x][y + 3]) * (self.user_weights[y] / average)))
                total += weightedScore[x][y]

            weightedScore[x].append(total)
            self.user_model_data[x][3:14] = weightedScore[x]
        self.user_model_data.sort(key=lambda x: x[-2], reverse=True)
        return weightedScore
    
    def export_user_model(self):
        fields = ['Index', 'Problem', 'Solution']
        fields.extend(self.new_metrics)
        fields.extend(['Combined Score', 'Category'])

        with open('./data/user_results.csv','w', newline = '', encoding = 'latin-1') as file:
            writer = csv.writer(file, fields)
            writer.writerow(fields)
            writer.writerows(self.user_model_data)

    def run_evaluator(self):
        cyclic_geese = '''    
  /$$$$$$                      /$$ /$$                  /$$$$$$                                         
 /$$__  $$                    | $$|__/                 /$$__  $$                                        
| $$  \__/ /$$   /$$  /$$$$$$$| $$ /$$  /$$$$$$$      | $$  \__/  /$$$$$$   /$$$$$$   /$$$$$$$  /$$$$$$ 
| $$      | $$  | $$ /$$_____/| $$| $$ /$$_____/      | $$ /$$$$ /$$__  $$ /$$__  $$ /$$_____/ /$$__  $$
| $$      | $$  | $$| $$      | $$| $$| $$            | $$|_  $$| $$$$$$$$| $$$$$$$$|  $$$$$$ | $$$$$$$$
| $$    $$| $$  | $$| $$      | $$| $$| $$            | $$  \ $$| $$_____/| $$_____/ \____  $$| $$_____/
|  $$$$$$/|  $$$$$$$|  $$$$$$$| $$| $$|  $$$$$$$      |  $$$$$$/|  $$$$$$$|  $$$$$$$ /$$$$$$$/|  $$$$$$$
 \______/  \____  $$ \_______/|__/|__/ \_______/       \______/  \_______/ \_______/|_______/  \_______/
           /$$  | $$                                                                                    
          |  $$$$$$/                                                                                    
           \______/ '''
        print(cyclic_geese)
        time.sleep(2)
        print("\nWelcome to the EcoPulse Idea Validator Tool!!")
        print("Our evaluator provides a baseline analysis of all of the ideas but also provides user-based analysis :)\n")
        time.sleep(4)
        print("Running Baseline Model...\n")
        self.baseline_model()
        time.sleep(4)
        print("Baseline Model run was succesful :)")
        print("The results of the baseline can be found in \'data/baseline_results.csv\'")
        print("Baseline results are sorted based on which idea we think are good\n")
        print("Here's a visualization of the categories of ideas in the dataset ;)\n")
        time.sleep(3)
        self.populate_categories()
        self.bar_visualization()
        time.sleep(4)
        print("Let's get into our user-based model")
        self.user_weights = self.user_model()
        self.evaluateAdditionalMetrics()
        time.sleep(4)
        weighted_scores = self.calculateScore()
        self.export_user_model()
        
        print("The results of the user model can be found in \'data/user_results.csv\'")
        print("The user model generates metrics based on the user's profile and evaluates ideas based on them\n")
        time.sleep(5)
        print("Now that we have the user model, would you like to filter the data based on a category?")
        print("Here are the categories: ")
        for category in self.categories.keys():
            print(category)
        print("\n")
        print("Enter y/n")
        filter = input()
        if filter == "y":
            print("\nEnter a category from the above list.")
            category = input()
            if category not in self.categories:
                print("Invalid category.")
            else:
                self.filter_categories(self.user_model_data, category)
                print(f"You can find the filtered dataset in \'data/filtered_{category}_.csv\'")
                time.sleep(2)
                print("\nThank you for using the EcoPulse idea evaluator!!!")

        else:
            print("\nThank you for using the EcoPulse idea evaluator!!!")
        
        honk_honk = '''  _                 _      _                 _    
                        | |               | |    | |               | |   
                        | |__   ___  _ __ | | __ | |__   ___  _ __ | | __
                        | '_ \ / _ \| '_ \| |/ / | '_ \ / _ \| '_ \| |/ /
                        | | | | (_) | | | |   <  | | | | (_) | | | |   < 
                        |_| |_|\___/|_| |_|_|\_\ |_| |_|\___/|_| |_|_|\_\
                                                  
                                                  '''


        print(honk_honk)
    

        