import socket
import io
import numpy as np
import pickle
import configparser
import os
import torch
import numpy as np
import trimesh
import sys
import cv2
import shutil

from pathlib import Path
from pytorch3d.io import save_obj
from semantify.utils._3dmm_utils import ThreeDMMUtils
from semantify.utils.models_factory import ModelsFactory
from semantify.utils.paths_utils import get_model_abs_path
from semantify.utils.general import get_model_to_eval
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate

from MICA.MICA import MICA, process_single, to_batch, deterministic


#Make Conversation Using Prompts
class ChatGPTConversation: 
    def __init__(self):
        
        config = configparser.ConfigParser()
        config.read("./keys.ini")
        openai_api_key = config['OPENAI']['OPENAI_API_KEY']
        os.environ.update({'OPENAI_API_KEY': openai_api_key})

        llm = OpenAI(temperature=0.9)
        template = """I have 4 basis of human expression : "smiling", "serious", "angry", "disgusted" this bases continously scaled between -1 and 1. For example with "smiling" basis with value of 0.7 and others with 0, it expresses smiling face. In opposite, with "smiling" basis with value of -1 and others with 0, it expresses strong sad face. Additionally combing multiple bases can express more diverse expressions. 

From now, you should answer predicted values of expressions you might feel for every user input. Answer in format smiling : value, serious : value, angry : value, disgusted : value
Do not include any other text to your answer.

Previous values:
{history}
User: {input}
Predicted Values:"""
        PROMPT = PromptTemplate(input_variables=["history", "input"], template=template)
        self.conversation = ConversationChain(
            prompt=PROMPT,
            llm=llm,
            memory=ConversationBufferMemory()
        )
        
    def __call__(self, prompt):
        self.conversation.memory.clear()
        text = self.conversation.predict(input=prompt)
        return text

class ReactToConversation:
    def __init__(self):
        #####SEMANTIFY CONFIGS#####
        mapper_path = get_model_abs_path("flame", "expression")
        num_coeffs = 10 
        
        self.model, self.labels = get_model_to_eval(mapper_path)
        self.threedmm_utils = ThreeDMMUtils()
        self.ignore_random_jaw = True
        self.device = torch.device("cuda")
        self.num_sentiments = 4
        self.model_kwargs = ModelsFactory("flame").get_default_params(with_face=True, num_coeffs=num_coeffs)
        self.model_kwargs["num_coeffs"] = num_coeffs
        self.model_kwargs["gender"] = "neutral"
        self.llm = ChatGPTConversation()
        self.shape_params = torch.zeros(300).unsqueeze(0)
        
        #####MICA CONFIGS#####
        pretrained='MICA/data/pretrained/mica.tar'
        self.mica, self.faces, self.app = MICA(pretrained, self.device)
        
        #####TCP(Local Python) CONNECTION#####
        port = 22222
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.bind(('0.0.0.0', port))  
        print("waiting for local server to send image")
        self.s.listen()
        self.client_socket, addr = self.s.accept()
        
        #####TCT(Unity) CONNECTION)########
        shutil.rmtree('./logs')
        os.makedirs('./logs')
        
    #Given input text make corresponding expression parameters
    def react_to(self, input_text, count=1):
        shape_params = self.shape_params
        prd = self.llm(input_text)
        pairs = prd.strip().split(',') #get values from gpt answer

        emotions = []
        for pair in pairs:
            _, value = pair.split(':')
            emotions.append((float(value.strip())*20) + 20)
        
        assert(self.num_sentiments == len(emotions))
        
        values = torch.tensor(emotions, dtype=torch.float32)[None]
        out = self.model(values.to(self.device)).cpu() #Predict real expression parameters
        expression_params = out
        np.save(f'./logs/{count}_expression_params.npy', out.detach().to('cpu').numpy())
        verts, faces = self.threedmm_utils.get_flame_model(expression_params=expression_params, shape_params=shape_params)
        
        # Save it to obj for test
        # os.makedirs("objs", exist_ok=True)
        # input_text = input_text.strip().replace(" ", "_")
        # obj_path = f"objs/{count}_{input_text}.obj"
        # save_obj(
        #     f=obj_path,
        #     verts=torch.tensor(verts).squeeze(),
        #     faces=torch.tensor(faces).squeeze()
        # )
        # print(f'saved at {obj_path}')
        
        return verts, faces
    
    #Get capture image from local
    def from_local(self):
        with self.client_socket:
            pickle_data = b""
            while True:
                data = self.client_socket.recv(4096)
                if data[-len(b"END"):] == b"END":
                    pickle_data += data[:-len(b"END")]
                    print("END MET")
                    break
                pickle_data += data
        
        received_image = pickle.loads(pickle_data)
        assert(received_image.shape[2] == 3)
        
        return received_image
    
    # MICA
    def convert_to_flame(self):
        input_img = self.from_local()
        mica = self.mica
        app = self.app
        arcface='MICA/demo/arcface'
        deterministic(42)
        with torch.no_grad():
            print("MICA started...")
            path = process_single(arcface, app, input_img, draw_bbox=False)
            images, arcface = to_batch(path)
            codedict = mica.encode(images, arcface)
            opdict = mica.decode(codedict)
            code = opdict['pred_shape_code']
            
            return code[0]
    
    def __call__(self):
        self.shape_params = self.convert_to_flame().unsqueeze(0)
        np.save("./logs/0_shape_params.npy", self.shape_params.to('cpu').numpy())
        count = 1
        while True:
            input_text = input("Say something to you : ")
            vert, faces = self.react_to(input_text, count)
            # self.to_local(vert, faces)
            count += 1
            if (input_text=='q'):
                break
        self.s.close()
            
            
if __name__ == "__main__":
    test = ReactToConversation()
    test()