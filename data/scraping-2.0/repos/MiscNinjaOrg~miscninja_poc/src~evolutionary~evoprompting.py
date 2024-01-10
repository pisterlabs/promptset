import openai
import concurrent.futures
import json
import os
import random
from datetime import datetime
import time
import numpy as np
from .climate_fever_dataset_temp import OurDataset
import streamlit as st

class EvoPrompting:
    def __init__(self, api_key, T, M, K, seed_files, prepend_file, data_dir, param_threshold):
        openai.api_key = api_key
        self.T = T  #how many total evolution steps to go through
        self.M = M  #how many children to generate in each evolution step
        self.K = K  #how many children are allowed to survive in each evolution step
        self.seed_files = seed_files
        self.prepend_file = prepend_file
        self.param_threshold = param_threshold
        self.population = []
        self.ds = OurDataset(os.path.join(data_dir, "full_data.json"), os.path.join(data_dir, "labels.torch"))
        self.curr_time = datetime.now()
        self.initialize_population()
        
    def initialize_population(self):
        # def read_seed_files(file_path):
        #     with open(file_path, "r") as file:
        #         return file.read()
        
        # self.prepend_code = read_seed_files(self.prepend_file)
        # seed_files = [f for f in os.listdir(self.seed_folder) if f.endswith('.py')]
        for seed_code in self.seed_files:
            model, metrics = self.exec_code(seed_code)
            if (model, metrics) == (0, 0):
                continue
            self.population.append((seed_code, model, metrics))
        
    def exec_code(self, code):
        def single_evaluation():
            print("Executing code segment")
            exec(self.prepend_file, globals())
            exec(code, globals())
            model = globals()['main']()
            return model
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        model = single_evaluation()
        if count_parameters(model) >= self.param_threshold:
            return (0, 0)
        metrics = self.eval_model(model)
        print("No. of Parameters: ", count_parameters(model))
        print("Model Score: ", metrics)
        return model, metrics
    
    #evaluate model without training here (just a classification forward pass)
    def eval_model(self, model):
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)
        loss = None
        for i in range(20):
            idx = random.randint(0, self.ds.__len__()-3000)
            x, y = self.ds.__getitem__(idx)
            y = y.long()
            outputs = model(x)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
        correct = 0
        for i in range(10):
            idx = random.randint(0, self.ds.__len__()-3000)
            x, y = self.ds.__getitem__(idx)
            y = y.long()
            outputs = model(x)
            outputs = torch.argmax(outputs, axis=-1)
            if outputs == y:
                correct += 1
        return correct/10
    
    def evaluate_children(self, child_architectures):
        for code in child_architectures:
            try:
                model, metrics = self.exec_code(code)
                self.population.append((code, model, metrics))
            except:
                continue
    
    # setup prompt and generate children
    def cross_mutation(self):
        while (datetime.now() - self.curr_time).seconds <= 60:
            time.sleep(2)
        self.curr_time = datetime.now()
        child_architectures = []
        for m in range(self.M):
            prompt = ""
            for arch in self.population: 
                prompt += "Code: " + arch[0] + "\n\n"
            prompt += "Code:"
            #print(prompt)
            messages = [#{"role":"system", "content":"You are a helpful assistant whose job is to generate better models based on the example given to you. Only generate valid python code."},
                       {"role":"user", "content":prompt}]
            response = openai.Completion.create(model="gpt-3.5-turbo-instruct", prompt=prompt, n=1, max_tokens=1600)
            # print(response["choices"][0]["text"].lstrip())
            child_architectures.append(response["choices"][0]["text"].lstrip())
        return child_architectures
    
    def cull_population(self):
        self.population = sorted(self.population, key=lambda x: x[-1], reverse=True)[:self.K]
        
    def evolve(self):
        for t in range(self.T):
            child_architectures = self.cross_mutation()
            self.evaluate_children(child_architectures)
            self.cull_population()
            print("Current Population Scores: ", [p[-1] for p in self.population])
        return self.population