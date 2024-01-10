from openai import OpenAI
import os
from dotenv import load_dotenv
import csv
import math
from numpy import *

from create_init_prompt import create_init_prompt
from improvement import improve_prompt
from image import image_val, create_image, save_image

# from mock import image_val, create_image, save_image, improve_prompt, create_init_prompt

parameter = {"expand_count": 3, "expand_width": 5, "image_num": 1, "max_iteration": 30}

class Node():
    def __init__(self, prompt, diff, cwd):
        self.origin_image = os.path.join("data", "image_" + str(parameter["image_num"]), "origin_" + str(parameter["image_num"]) + ".jpg")
        if not os.path.exists(self.origin_image):
            raise Exception("The image does not exist: " + self.origin_image)
        
        self.directory = cwd
        self.prompt = prompt
        self.diff = diff
        
        self.images = []
        self.scores = []
        
        self.w = 0
        self.n = 0
        self.child_nodes = None

    def expand(self):
        # scoreの一番低いindexを取得する
        min_score_index = argmin(self.scores)
        # そのindexのimageを取得する
        min_score_image = self.images[min_score_index]
            
        
        self.child_nodes = []
        for i in range(parameter["expand_width"]):
            diff, new_prompt = improve_prompt(self.origin_image, min_score_image, self.prompt)
            new_dir = os.path.join(self.directory, "node_" + str(i+1))
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            
            self.child_nodes.append(Node(new_prompt, diff, new_dir))
            
    def next_child_node(self):
        for child_node in self.child_nodes:
            if child_node.n == 0:
                return child_node
        
        t = 0
        for child_node in self.child_nodes:
            t += child_node.n
        ucb1_values = []
        for child_node in self.child_nodes:
            ucb1_values.append(child_node.w / child_node.n + 2 * (2 * math.log(t) / child_node.n) ** 0.5)
            
        return self.child_nodes[argmin(ucb1_values)]
            
    def evaluate(self):
        if not self.child_nodes:
            new_image_http = create_image(self.prompt)
            new_image = os.path.join(self.directory, "image_" + str(parameter["image_num"]) + "_" + str(len(self.images)) + ".jpg")
            save_image(new_image, new_image_http)
            new_score = image_val(self.origin_image, new_image)
            
            self.images.append(new_image)
            self.scores.append(new_score)
            
            self.w += new_score
            self.n += 1
            
            if self.n == parameter["expand_count"]:
                self.expand()
            return new_score
        
        else:
            new_score = self.next_child_node().evaluate()
            
            self.w += new_score
            self.n += 1
            return new_score
    
    # 集計
    def aggregation(self):
        print(self.directory)
        
        if self.n == 0:
            return 1000000, "", "", ""
        
        file_path = os.path.join(self.directory, "evaluation.csv")
        id = 0

        with open(file_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Prompt", "Diff"])
            writer.writerow([self.prompt, self.diff])
            writer.writerow(["ID", "Image", "Evaluation"])
            for score, image in zip(self.scores, self.images):
                writer.writerow([id, score, image])
                id += 1
                
        file.close()
        
        if not self.child_nodes:
            return self.w / self.n, self.prompt, self.diff , self.images[argmin(self.scores)]
        
        else:
            min_score = self.w / self.n
            min_prompt = self.prompt
            min_diff = self.diff
            min_image = self.images[argmin(self.scores)]
            for child_node in self.child_nodes:
                score, prompt, diff, image = child_node.aggregation()
                if score < min_score:
                    min_score = score
                    min_prompt = prompt
                    min_diff = diff
                    min_image = image
            
            return min_score, min_prompt, min_diff, min_image
        
if __name__ == "__main__":
    cwd = os.path.join("data", "image_" + str(parameter["image_num"]), "mcts_1")
    if not os.path.exists(cwd):
        os.makedirs(cwd)
    
    origin_image = os.path.join("data", "image_" + str(parameter["image_num"]), "origin_" + str(parameter["image_num"]) + ".jpg")
    root_node = Node(create_init_prompt(origin_image), "", cwd)
    image = create_image(root_node.prompt)
    image_path = os.path.join(cwd, "image_0.jpg")
    save_image(image_path, image)
    root_node.images.append(image_path)
    root_node.scores.append(image_val(root_node.origin_image, root_node.images[0]))
    root_node.expand()
    
    for i in range(parameter["max_iteration"]):
        root_node.evaluate()
        print(f"iteration: {i+1}/{parameter['max_iteration']}")
        
    score, prompt, _, image = root_node.aggregation()
    print("score: " + str(score))
    print("prompt: " + prompt)
    print("image: " + image)
    