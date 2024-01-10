import os
from langchain.embeddings.openai import OpenAIEmbeddings
import openai
import faiss
from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from utils import file_reader, parse_aim_resumes
import dotenv
from tqdm import tqdm
dotenv.load_dotenv()
openai.api_key = os.environ.get('OPENAI_API_KEY')
embeder = OpenAIEmbeddings()


def main(path: str, vec_path='vectorstore2'):
    """
    The profiles directory is a series of directories. Each has two files
    (one docx and one of many types). If it starts with 'AIM P' then it's
    an AIM profile; otherwise, it's the user's resume.

    The goal is to index these files into a single vector + metadata.
    """
    make_empty_db(vec_path)
    missed_resumes = []
    for dir_path in tqdm(os.listdir(path)):
        aim_profile = ''
        resume = ''
        full_dir_path = os.path.join(path, dir_path)
        for file in os.listdir(full_dir_path):
            if file.startswith('~'):
                continue
            file_path = os.path.join(full_dir_path, file)
            mode = 'r' if file.endswith('.txt') else 'rb'
            with open(file_path, mode) as f:
                try:
                    if file.startswith('AIM P'):
                        aim_profile = file_reader(f)
                    else:
                        resume = file_reader(f)
                except Exception as e:
                    print(f'Error reading {file_path}: {e}')
                    missed_resumes.append(file_path)
                    continue
        try:
            summary, skills, experience, education, certifications, awards = parse_aim_resumes(aim_profile)
        except Exception as e:
            print(f'Error parsing {dir_path}: {e}')
            missed_resumes.append(dir_path)
            continue
        if not resume:
            missed_resumes.append(dir_path)
            continue
        try:
            embed_example(resume, summary, skills, experience, education, certifications, awards, dir_path,
                          save_path=vec_path)
        except Exception as e:
            print(f'Error embedding {dir_path}: {e}')
            missed_resumes.append(dir_path)
        print(f'Missed resumes ({len(missed_resumes)}): {missed_resumes}')


def embed_example(resume, summary, skills_and_tech, professional_experience, education, certifications, awards, path, save_path='vectorstore'):
    """
    Save the index with the resume + the metadata so it can be extracted later.
    """
    meta_data = {
        "resume": resume,
        "summary": summary,
        "skills_and_tech": skills_and_tech,
        'professional_experience': professional_experience,
        'education': education,
        'certifications': certifications,
        'awards': awards
    }

    # embedding = embeder.embed_query(resume)
    # load db
    db = FAISS.load_local(save_path, OpenAIEmbeddings())
    db.add_texts(texts=[resume], metadatas=[meta_data], ids=[path])
    db.save_local(save_path)


def make_empty_db(path='vectorstore/'):
    """
    Create an empty db for the vector store.
    """
    embeddings = OpenAIEmbeddings()
    # Create empty index
    index = faiss.IndexFlatL2(1536)

    # Empty docstore
    docstore = InMemoryDocstore({})

    # Empty ID mapping
    index_to_docstore_id = {}

    db = FAISS(embeddings, index, docstore, index_to_docstore_id)
    db.save_local(path)


if __name__ == '__main__':
    main(path='profiles')
