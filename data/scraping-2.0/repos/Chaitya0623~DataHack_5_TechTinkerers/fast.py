# uvicorn [filename]:app --reload

from transformers import pipeline
from fastapi import FastAPI, Body, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
from pydantic import BaseModel
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
import os
import PyPDF2
import re
import pickle
import torch
import gc
from langchain.llms import OpenAI
llm = OpenAI(openai_api_key="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")
import warnings
warnings.filterwarnings('ignore')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)
class SentenceRequest(BaseModel):
    skills: list

class SentenceResponse(BaseModel):
    projects: str

# os.environ["OPENAI_API_KEY"] = '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'
# embeddings = OpenAIEmbeddings()
# db = FAISS.load_local('faiss_index', embeddings)

# @app.post("/wikipedia_ts", response_model=SentenceResponse)
# async def truth_similarity(request: SentenceRequest = Body(...)):
#     query = request.sentence
#     docs_and_scores = db.similarity_search_with_score(query, 2)
#     output = {}
#     for i in range(len(docs_and_scores)):
#         doc_dict = {}
#         data = docs_and_scores[i]
#         document, score = data
#         metadata = document.metadata
#         page_content = document.page_content
#         created_at = metadata.get('created_at', None)
#         links = metadata.get('links', None)
#         topic = metadata.get('topic', None)
#         doc_dict['page_content'] = page_content
#         doc_dict['created_at'] = created_at
#         doc_dict['links'] = links
#         doc_dict['topic'] = topic
#         doc_dict['score'] = float(score)
#         output[i] = doc_dict
#     response = SentenceResponse(truths=output)
#     return response

with open('mnli.pkl', 'rb') as file:
    classifier = pickle.load(file)
candidate_labels =  ['Frontend (HTML, CSS, JavaScript, React, TailwindCSS, ChakraUI, ThreeJs, LaTex,Figma, Sketch, Adobe XD, Animation )',
  'Backend (NodeJS, Django, Express, MongoDB, MySQL, Go, REST_Framework)',
  'Machine Learning (Numpy, Pandas, Keras, Seaborn, Matplotlib, Scikit-Learn, Tensorflow)',
  'Computer Vision (Yolo, RCNN, Haarcascade, mediapipe, pytorch, OCR,Image Analysis, Object Detection, and Video Processing; Proficient in Deep Learning Frameworks,PyTorch, TensorFlow,CNNs)',
  'Natural Language Processing (Langchains, haystacks, Large Language Models (LLMs), Transformers)',
  'Cloud Computing (Amazon Web Services (AWS), Azure, Docker, Terraforms, Kubernetes, Google Cloud Platform, Elastic, Oracle Cloud)',
  'Blockchain (Solidity, vyper, Proof of work, Go, Rust, motoko)',
  'Programming (C, C++, Java, Python)',
  'Data Science (Google Looker Studio, PowerBi, Hadoop, R, MatLab, Julia, Scala)',
  'Graphic Designer (Branding, Print Design, Adobe, Photoshop, Illustrator, UI/UX Design)'
]

@app.post("/user_skill")
async def skill_calculation(file: UploadFile = File(...)):
    gc.collect()
    # Process the uploaded file
    def extract_text_from_pdf(pdf_file):
        pdf = PyPDF2.PdfFileReader(pdf_file)

        # Initialize an empty string to store the text
        pdf_text = ""

        # Loop through each page and extract text
        for page_num in range(pdf.numPages):
            page = pdf.getPage(page_num)
            page_text = page.extractText()
            pdf_text += page_text

        return pdf_text
    
    pdf_text = extract_text_from_pdf(file.file)

    # Remove special characters and extra spaces
    cleaned_text = re.sub(r'[^\w\s/:-]', '', pdf_text)

    # Replace multiple spaces and line breaks with a single space
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

    # Strip leading and trailing spaces
    cleaned_text = cleaned_text.strip()
    output = classifier(cleaned_text, candidate_labels, multi_label=False)
    return {'labels': output['labels'], 'score': output['scores']}


from langchain import PromptTemplate
from langchain.chains import LLMChain

generate_example_template = """

% INSTRUCTIONS
You are a person who recommends people to upskill themselves.

% TEXTUAL QUESTION
{label}

% YOUR TASK
Suggest projects that the person should do, to improve, based on their current skillset.
"""
  
@app.post('./project_rec', response_model=SentenceResponse)
async def project_recommendation(request: SentenceRequest = Body(...)):
    label = request.skills
    print(label)
    prompt = PromptTemplate.from_template(generate_example_template)
    formatted_prompt = prompt.format(label=label)

    chain = LLMChain(llm=llm, prompt=prompt)
    example = chain.run(formatted_prompt)
    example = example.replace('\n','')
    print(example)
    return example
# with open('dolly.pkl', 'rb') as file:
#     generate_text = pickle.load(file)
# @app.post("/project_rec", response_model=SentenceResponse)
# async def project_recommendation(request: SentenceRequest = Body(...)):
#     gc.collect()
#     query = request.skills
#     label = ['OpenCV', 'AI', 'Robot Operating Systems']
#     res = generate_text(f"Suggest me a project, considering my current skillset includes {label}.")
#     response = res[0]['generated_text'].replace('\n','')
#     return response

# http://127.0.0.1:8000/docs open this site and try it out