import os
import json
# from langchain.embeddings import OpenAIEmbeddings

# embeddings_model = OpenAIEmbeddings(openai_api_key="...")

# open all files from the data/ directory, and collect the data, return a list
def open_files(folder='data'):
    fol = input('Enter folder name: ')
    if fol:
        folder = fol
    files = []
    for filename in os.listdir(folder):
        if filename.endswith('.txt'):
            with open(f'{folder}' + filename, 'r') as f:
                files.append(f.read())
    return files

# for each file, split by ~ and return a list of lists
def split_files(files):
    split_files = []
    for file in files:
        split_files.extend(file.split('!@#'))
    return split_files

# for each passed split file, extract education and experience by splitting by double #
def extract_education_and_experience(split_files):
    education_experience = []
    for file in split_files:
        split_file = file.split('##')
        education_experience.append(split_file)
    return education_experience

def final_chunks(edu_exp_element):
    education = edu_exp_element[0]
    experience = edu_exp_element[1]
    education_chunks = education.split('#')
    experience_chunks = experience.split('#')
    return education_chunks, experience_chunks

def store_data(education_experience, filename='data.json'):
    data = {}
    for i, element in enumerate(education_experience):
        education_chunks, experience_chunks = final_chunks(element)
        data[i] = {
            "edu_chunks": education_chunks,
            "exp_chunks": experience_chunks
        }
    flname = input('Enter file name: ')
    if flname:
        filename = flname
    with open(f'data/{filename}', 'w') as f:
        json.dump(data, f, indent=4)

def chunking_pipeline(folder='data', filename='data.json'):
    files = open_files(folder)
    split_files = split_files(files)
    education_experience = extract_education_and_experience(split_files)
    store_data(education_experience, filename)

# def embed():
#     files = open_files()
#     split_files = split_files(files)
#     education_experience = extract_education_and_experience(split_files)
#     education_chunks, experience_chunks = final_chunks(education_experience[0])
#     education_embeddings = get_embedding(education_chunks)
#     experience_embeddings = get_embedding(experience_chunks)
#     return education_embeddings, experience_embeddings

# if __name__ == '__main__':
#     files = open_files()
#     split_files = split_files(files)
#     education_experience = extract_education_and_experience(split_files)
#     store_data(education_experience)
    # education_chunks, experience_chunks = final_chunks(education_experience[0])