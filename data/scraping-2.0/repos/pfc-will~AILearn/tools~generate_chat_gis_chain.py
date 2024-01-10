'''
Write a tool to recursively a directory and its subdirectories to return all file list with a file extension.
'''

import os
import shutil

server_path = os.path.dirname(os.path.dirname(__file__))
output_path = os.path.join(server_path, 'chat_gis_chain')

def get_file_list(path, extension):
    file_list = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(extension):
                file_list.append(os.path.join(root, file))
    return file_list

if __name__ == '__main__':
    '''
    Change the path to your own
    '''
    path = '/Users/wangyawei/Documents/work/ChatGIS/env/lib/python3.8/site-packages/langchain'
    extension = '.py'
    files = get_file_list(path, extension)
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.mkdir(output_path)
    for file in files:
        file = file.replace(f'{path}{os.path.sep}', '')
        # Create a text file and its directory if it does not exist
        file_path = os.path.join(output_path, file)
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        with open(file_path, 'w') as f:
            if file == '__init__.py':
                f.write(f'from langchain import *\n')
            elif file.endswith('__init__.py'):
                f.write(f'from langchain.{file.replace(f"{os.path.sep}__init__.py", "").replace(os.path.sep, ".")} import *\n')
            else:
                f.write(f'from langchain.{file.replace(os.path.sep, ".")[:-3]} import *\n')


