import boto3
import pathlib
import os
import re
import base64
import json
import numpy as np

from langchain.llms import Bedrock
from langchain.embeddings import BedrockEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms.utils import enforce_stop_tokens

RELEVANCE_TEMPLATE = """\n\nHuman: Generate question for the given answer.\n\nAssistant:Okay, give me an answer, and I will generate a question.
\nHuman:Answer:\nThe PSLV-C56 mission is scheduled to be launched on Sunday, 30 July 2023 at 06:30 IST / 01:00 UTC. It will be launched from the Satish Dhawan Space Centre, Sriharikota, Andhra Pradesh, India 
\nAssistant:Question:\nWhen is the scheduled launch date and time for the PSLV-C56 mission, and where will it be launched from?
\nHuman:Answer:\n{answer}
\nAssistant:Question:\n
""" 

EVALUATOR = PromptTemplate(template=RELEVANCE_TEMPLATE, input_variables=["answer"])

# claculate how similar is the original question vs LLM generated questions
# using cosine similarity
def calculate_similarity(question, generated_questions, embeddings):
    
    question_vec = np.asarray(embeddings.embed_query(question)).reshape(1, -1)
    gen_question_vec = np.asarray(
        embeddings.embed_documents(generated_questions)
    )
    norm = np.linalg.norm(gen_question_vec, axis=1) * np.linalg.norm(
        question_vec, axis=1
    )
    return (
        np.dot(gen_question_vec, question_vec.T).reshape(
            -1,
        )
        / norm
    )

def base64_to_string(base64_string):
    base64_bytes = base64_string.encode('ascii')
    string_bytes = base64.b64decode(base64_bytes) 
    return string_bytes.decode('utf-8')

def extract_questions(text):
    pattern = r"### Question\n(.*?)\n### Context"
    match = re.search(pattern, text, re.DOTALL)
    if match is None:
        return ""
    return match.group(1)

def extract_answers(text):
    pattern = r"\[\/INST\](.*)"
    match = re.search(pattern, text)
    if match is None:
        return ""
    return match.group(1)   

def extract_contexts(text):
    pattern = r"### Context\n(.*)\[/INST]"
    match = re.search(pattern, text, re.DOTALL)
    if match is None:
        return ""
    return match.group(1)

# Helper function to extract question and answer from dataset
def extract_qac(input_data, output_data):
    question = extract_questions(json.loads(base64_to_string(input_data))["text"])
    print("Question: ", question)
    context = extract_contexts(json.loads(base64_to_string(input_data))["text"])
    print("Context: ", context)
    generated_text = json.loads(base64_to_string(output_data))["outputs"][0]["generated_text"]
    answer = extract_answers(generated_text)
    print("Answer: ", answer)
    return question, answer, context

def main():    
    
    # Load dataset
    questions, answers = [], []
    infer_dir = os.environ['dataset_source']
    
    jsonl_files = pathlib.Path(infer_dir).rglob('*.jsonl')
    
    if not jsonl_files:
        print('No *.jsonl files found.') 
        exit()

    for filepath in jsonl_files:

        with open(filepath.absolute(), 'r') as f:
            for line in f:
                jsonl = json.loads(line)
                input_data = jsonl['captureData']['endpointInput']['data']
                output_data = jsonl['captureData']['endpointOutput']['data']
                q, a, c = extract_qac(input_data, output_data)
                if q != "" and a != "":
                    questions.append(q)
                    answers.append(a)

    #print(questions)
    #print(answers)
    
    # Initialize LLMs            
    llm = Bedrock(
        model_id="anthropic.claude-v2",
        model_kwargs={"max_tokens_to_sample": 200,
                    "temperature": 0},
        client=boto3.client("bedrock-runtime", region_name='us-west-2'),
    )
    
    llm_chain = LLMChain(llm=llm, prompt=EVALUATOR)
        
    embeddings= BedrockEmbeddings(
        client=boto3.client("bedrock-runtime", region_name='us-west-2'),
    )

    scores = []

    for q, a in zip(questions, answers):
        results = []
        for i in range(5):
            results.append(llm_chain.run(answer=a).strip())
        cosine_sim = calculate_similarity(q, results, embeddings)
        scores.append(cosine_sim.mean())

    print(f"average relevancy score: {np.mean(scores)*100} /100")

    output = {
        "llm_metrics": {
            "answer_relevancy": {"value": np.mean(scores), "standard_deviation": np.std(scores)},
        },
    }
    
    os.makedirs(os.environ['output_path'], exist_ok=True)
    with open(f"{os.environ['output_path']}/results.json", 'w+') as f:
        json.dump(output, f)


if __name__ == '__main__':
    
    main()
