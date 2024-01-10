import os
import openai

openai.log = "debug"
# made available for docker
dataFolder = "data"
def get_data_folder_full_path():
    currentPath = os.path.dirname(os.path.realpath(__file__))
    datafolderFullPath = os.path.join(currentPath, dataFolder)
    return datafolderFullPath