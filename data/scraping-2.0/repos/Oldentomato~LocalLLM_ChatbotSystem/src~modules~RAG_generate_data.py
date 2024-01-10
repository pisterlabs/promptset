import json

from llama_index import SimpleDirectoryReader
from llama_index.node_parser import SimpleNodeParser
from llama_index.schema import MetadataMode
import re
import uuid

from llama_index.llms import OpenAI
from llama_index.schema import MetadataMode
from tqdm.notebook import tqdm
from config import OPENAI_API_KEY


TRAIN_FILES = ['../data/corona.pdf']
VAL_FILES = ['../data/corona.pdf']

TRAIN_CORPUS_FPATH = '../data/train_corpus.json'
VAL_CORPUS_FPATH = '../data/val_corpus.json'

def load_corpus(files, verbose=False):
    if verbose:
        print(f"Loading files {files}")

    reader = SimpleDirectoryReader(input_files=files)
    docs = reader.load_data()
    if verbose:
        print(f'Loaded {len(docs)} docs')
    
    parser = SimpleNodeParser.from_defaults()
    nodes = parser.get_nodes_from_documents(docs, show_progress=verbose)

    if verbose:
        print(f'Parsed {len(nodes)} nodes')

    corpus = {node.node_id: node.get_content(metadata_mode=MetadataMode.NONE) for node in nodes}
    return corpus


train_corpus = load_corpus(TRAIN_FILES, verbose=True)
val_corpus = load_corpus(VAL_FILES, verbose=True)

with open(TRAIN_CORPUS_FPATH, 'w+', encoding='utf-8') as f:
    json.dump(train_corpus, f, ensure_ascii=False)

with open(VAL_CORPUS_FPATH, 'w+', encoding='utf-8') as f:
    json.dump(val_corpus, f, ensure_ascii=False)

TRAIN_QUERIES_FPATH = '../data/train_queries.json'
TRAIN_RELEVANT_DOCS_FPATH = '../data/train_relevant_docs.json'

VAL_QUERIES_FPATH = '../data/val_queries.json'
VAL_RELEVANT_DOCS_FPATH = '../data/val_relevant_docs.json'


with open(TRAIN_CORPUS_FPATH, 'r+', encoding='utf-8') as f:
    train_corpus = json.load(f)

with open(VAL_CORPUS_FPATH, 'r+') as f:
    val_corpus = json.load(f)

"""\
    Context information is below.
    
    ---------------------
    {context_str}
    ---------------------
    
    Given the context information and not prior knowledge.
    generate only questions based on the below query.
    
    You are a Teacher/ Professor. Your task is to setup \
    {num_questions_per_chunk} questions for an upcoming \
    quiz/examination. The questions should be diverse in nature \
    across the document. Restrict the questions to the \
    context information provided."
    """

def generate_queries(
    corpus,
    num_questions_per_chunk=2,
    prompt_template=None,
    verbose=False,
):
    """
    Automatically generate hypothetical questions that could be answered with
    doc in the corpus.
    """
    llm = OpenAI(model='gpt-3.5-turbo', api_key=OPENAI_API_KEY)

    prompt_template = prompt_template or """\
    컨텍스트 정보는 아래와 같습니다.
    
    ---------------------
    {context_str}
    ---------------------
    
    사전 지식이 아닌 컨텍스트 정보를 제공합니다.
    아래 쿼리를 기반으로 질문만 생성합니다.
    
    당신은 교사/교수입니다. 당신의 임무는 컨텍스트 정보가 주어지면\
    {num_questions_per_chunk} 개의 질문들을 생성하여 퀴즈/시험 형식으로 구성해주는 것입니다. \
    문제는 본질적으로 문서 전반에 걸쳐 다양해야 합니다. \
    제공된 컨텍스트 정보로 질문을 제한합니다.
    """

    queries = {}
    relevant_docs = {}
    for node_id, text in tqdm(corpus.items()):
        query = prompt_template.format(context_str=text, num_questions_per_chunk=num_questions_per_chunk)
        response = llm.complete(query)
 
        result = str(response).strip().split("\n")
        questions = [
            re.sub(r"^\d+[\).\s]", "", question).strip() for question in result
        ]
        questions = [question for question in questions if len(question) > 0]
        
        for question in questions:
            question_id = str(uuid.uuid4())
            queries[question_id] = question
            relevant_docs[question_id] = [node_id]
    return queries, relevant_docs

train_queries, train_relevant_docs = generate_queries(train_corpus)

val_queries, val_relevant_docs = generate_queries(val_corpus)

with open(TRAIN_QUERIES_FPATH, 'w+', encoding='utf-8') as f:
    json.dump(train_queries, f, ensure_ascii=False)

with open(TRAIN_RELEVANT_DOCS_FPATH, 'w+', encoding='utf-8') as f:
    json.dump(train_relevant_docs, f, ensure_ascii=False)

with open(VAL_QUERIES_FPATH, 'w+', encoding='utf-8') as f:
    json.dump(val_queries, f, ensure_ascii=False)

with open(VAL_RELEVANT_DOCS_FPATH, 'w+', encoding='utf-8') as f:
    json.dump(val_relevant_docs, f, ensure_ascii=False)

TRAIN_DATASET_FPATH = '../data/train_dataset.json'
VAL_DATASET_FPATH = '../data/val_dataset.json'

train_dataset = {
    'queries': train_queries,
    'corpus': train_corpus,
    'relevant_docs': train_relevant_docs,
}

val_dataset = {
    'queries': val_queries,
    'corpus': val_corpus,
    'relevant_docs': val_relevant_docs,
}

with open(TRAIN_DATASET_FPATH, 'w+', encoding='utf-8') as f:
    json.dump(train_dataset, f, ensure_ascii=False)

with open(VAL_DATASET_FPATH, 'w+', encoding='utf-8') as f:
    json.dump(val_dataset, f, ensure_ascii=False)