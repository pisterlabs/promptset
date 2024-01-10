#set key with export OPENAI_API_KEY="..."

import os
import openai

from build_prompt import get_prompt
from settings import DATA_DIR, LM_model

def get_answer(
        query: str, 
        text_dir: str,
        id_list: list[str]) -> str:
    """From a question asked by the user, generate the answer

    Args:
        query (str): Question asked by the user
        text_dir (str): Documents directory
        id_list (list[str]): List of relevant docs

    Returns:
        str: Answer generated with the LLM
    """         
    docs,seperator_list = get_texts_from_ids(id_list=id_list,
                                             text_dir=text_dir)
        
    if len(seperator_list) == (len(docs) + 1):
        meta_data = seperator_list.pop(0)
        seperator_list[0] = meta_data + seperator_list[0]

    #assemble the prompt
    this_prompt = get_prompt(query, docs, seperator_list)
    if (this_prompt == None):
        return None

    #call openai to obtain a response
    if LM_model == "gpt-3.5-turbo":
        response = openai.ChatCompletion.create(
            model = LM_model,
            messages=[ {"role": "system", "content": "You are a helpful assistant."},
                       {"role": "user", "content": this_prompt}],
            temperature = 0,
            max_tokens = 500,
            )
        result = response['choices'][0]['message']['content']
    else:
        response = openai.Completion.create(
            model = "text-davinci-003",
            prompt = this_prompt,
            temperature = 0,
            max_tokens = 500,
        )
        result = response['choices'][0]['text']

    return result

def get_texts_from_ids(id_list: list[str],
                       text_dir: str) -> tuple[list[str], list[str]]:
    """Returns a tuple with the texts from the text chunks with the specified 
    IDs and a list with the information from the according meta files. If no 
    meta file exists, the second list contains 'Here is a new text' as an
    entry. The second list can be used as a seperator list for the prompt 
    building.
    
    Args:
        id_list (list[str]): List with the IDs of the sleected chunks.
        text_dir (str): Documents directory.
        
    Returns:
        list[str]: A list that contains the texts from the chunks with the 
            selected IDs.
        list[str]: A list that contains the meta information from the 
            corresponding meta files if they exist.
        """
    docs = []
    seperator_list = []

    # First read in the chunks given by the id_list from the given directory.
    # Either the file exists as in the directory or it is located in a sub 
    # directory, file name without chunk id.

    # add meta data that holds for all files in the directory if exists
    meta_file = os.path.join(text_dir, 'meta.txt')  # meta file for all texts
    if os.path.isfile(meta_file) and os.access(meta_file, os.R_OK):
        with open(meta_file, 'r', encoding="utf-8") as file:
                meta_data = file.read()
        seperator_list.append(f'Here is some meta information that hold for '
                              + f'all texts that are given to answer the '
                              + f'question:\n{meta_data}\n\n')

    for id in id_list:
        file_path = os.path.join(text_dir, id + ".txt")
        dir_path = os.path.join(text_dir, id.split('.')[0]) # sub dir
        file_in_sub_dir = False
        # search for a sub directory with chunks:
        if(not os.path.isfile(file_path) and os.path.isdir(dir_path)):
            file_path = os.path.join(dir_path, id + ".txt")
            file_in_sub_dir = True  # to get the location of the meta file

        if not os.access(file_path, os.R_OK):
            print(f'ERROR: Could not find or access the file {id}.txt '
                  + f' directly or in a sub directory {dir_path}.')
            exit(1)
                
        this_chunk = open(file_path, "r", encoding="UTF-8")
        docs.append(this_chunk.read())
        this_chunk.close()

        # Create seperator for the chunk using the meta file data if a meta 
        # file exist:
        if file_in_sub_dir == True:
            meta_file = os.path.join(dir_path, id.split('.')[0] + '.meta.txt')
        else:
            meta_file = os.path.join(text_dir, id.split('.')[0] + '.meta.txt')

        if os.path.isfile(meta_file) and os.access(meta_file, os.R_OK):
            # read in the meta data into seperators
            with open(meta_file, 'r') as file:
                meta_data = file.read()
            seperator_list.append(f'Here are some meta information about the '
                                  + f'following text:\n{meta_data}.\n The '
                                  + 'corresponding text:')
        else:
            # ignore meta file, continue without meta information
            seperator_list.append('This is a new text:')
    
    return docs, seperator_list
    
def main():
    testq = "What are common research interests of these scientists?"
    testdir = os.path.join(DATA_DIR, "publications")
    testlist = ['2','4']
    testanswer = get_answer(query=testq, text_dir=testdir, id_list=testlist)
    print(testanswer)

if __name__ == "__main__":
    main()
