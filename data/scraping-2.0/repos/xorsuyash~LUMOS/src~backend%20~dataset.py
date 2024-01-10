import os
import sys 
import tqdm 
from langchain.text_splitter import CharacterTextSplitter

#function which takes in the path and then it run parse the files present in the folder 
# Subject - Units and then topics 
# if its a book then whole book in general 
# web parser 
# pdf parser 
# book parser 
# parse karne ke baad json format me store kar denge 
#text data for gfg 

class VocabuLary:
    def __init__():
        pass 
    def ch_to_index():
        pass 
    def inedex_to_character():
        pass 
    
    
    
    

class TextDataset:
    
    def __init__(self,parent_folder):
        self.parent_folder=parent_folder
        
        
    def path_loader(self):
        
        return self._find_files(self.parent_folder)
     
    
    def _find_files(self,root_dir):
        file_list=[]
        for root, _, files in os.walk(root_dir):
            for file in files:
                file_list.append(os.path.join(root, file))
        
        return file_list 
        
    
    def text_splitter(self):
        file_list=self.path_loader()
        text_splitter=CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    
        )
        text='<SOS>'
        for file in file_list:
            with open(file,'r') as f:
                raw_text=f.read()
            text=text+' '+raw_text
        chunks = text_splitter.split_text(text)
        return chunks 
    
    def text_loader(self):
        return self.text_splitter()
        
    def vocabulary(self):
        pass 
    
        
        

    
    