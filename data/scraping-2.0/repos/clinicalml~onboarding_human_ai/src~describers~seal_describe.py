import logging
import os

import numpy as np
import openai
import torch


class SEALDescribe:
    def __init__(self, descriptions, cluster_labels, task, open_ai_key):
        self.descriptions = descriptions
        self.cluster_labels = cluster_labels
        self.task = task
        openai.api_key = open_ai_key
        self.descriptions_gen = []

    def get_completion(self, prompt):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
                # {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
                # {"role": "user", "content": "Where was it played?"}
            ],
        )
        logging.info("Called OPENAI API")
        return response["choices"][0]["message"]["content"]

    def build_prompt(self, content, task):
        instruction = (
            "In this task , we `ll assign a short and precise label to a group of documents based on the topics or concepts most relevant to these documents . The documents are all subsets of a"
            + task
            + "dataset ."
        )
        examples = "\n - ".join(content)
        prompt = instruction + "- " + examples + "\n Group label :"
        return prompt

    def describe_region(self, cluster_selected):
        inside_caption_set_idx = []
        for j in range(len(self.cluster_labels)):
            if self.cluster_labels[j] == cluster_selected:
                inside_caption_set_idx.append(j)

        # sample 20  points from inside_caption_set_idx
        selected_idx = np.random.choice(
            inside_caption_set_idx, min(20, len(inside_caption_set_idx)), replace=False
        )
        # get their captions
        selected_points = [self.descriptions[i] for i in selected_idx]

        # get description
        description = self.get_completion(self.build_prompt(selected_points, self.task))
        self.descriptions_gen.append(description)
        logging.info(f" Description: {description} ")
        return description, selected_points
