import asyncio
import json
import math
import os
import random
import re
import shutil
import uuid

import assemblyai as aai
import pandas as pd
import requests
import streamlit as st
import weaviate
from langchain import hub, LLMChain
from langchain.document_loaders import OnlinePDFLoader
from langchain.document_loaders import YoutubeAudioLoader
from langchain.embeddings import ClarifaiEmbeddings
from langchain.llms import Clarifai
from llama_index import Document, LangchainEmbedding, ServiceContext, VectorStoreIndex
from llama_index.llms import LangChainLLM
from llama_index.node_parser import SimpleNodeParser
from llama_index.node_parser.extractors import MetadataExtractor, QuestionsAnsweredExtractor
from llama_index.vector_stores import WeaviateVectorStore

PROJECT_NAME = 'quiz-wiz'

auth_config = weaviate.AuthApiKey(api_key=os.getenv('WEAVIATE_API_KEY'))

CLF_OPENAI_USER_ID = 'openai'
CLF_CHAT_COMPLETION_APP_ID = 'chat-completion'
CLF_EMBED_APP_ID = 'embed'
CLF_GPT4_MODEL_ID = 'GPT-4'
CLF_EMBED_MODEL_ID = 'text-embedding-ada'
CLF_GPT35_MODEL_ID = 'GPT-3_5-turbo'

WEAVIATE_CLASS_PREFIX = "StreamlitDocument"

CLARIFAI_PAT = st.secrets['CLARIFAI_PAT']

QA_GEN_PROMPT = hub.pull("aaalexlit/context-based-question-generation").template
QUIZ_GEN_PROMPT = hub.pull("aaalexlit/quizz-creator")

aai.settings.api_key = st.secrets['ASSEMBLYAI_API_KEY']

# Initialize a Clarifai LLM
langchain_llm = Clarifai(
    pat=CLARIFAI_PAT,
    user_id=CLF_OPENAI_USER_ID,
    app_id=CLF_CHAT_COMPLETION_APP_ID,
    model_id=CLF_GPT4_MODEL_ID,
)
llamaindex_llm = LangChainLLM(langchain_llm)

qa_llm = LangChainLLM(Clarifai(
    pat=CLARIFAI_PAT,
    user_id=CLF_OPENAI_USER_ID,
    app_id=CLF_CHAT_COMPLETION_APP_ID,
    model_id=CLF_GPT35_MODEL_ID,
))

# Initialize a Clarifai embedding model
embeddings = ClarifaiEmbeddings(
    pat=CLARIFAI_PAT,
    user_id=CLF_OPENAI_USER_ID,
    app_id=CLF_EMBED_APP_ID,
    model_id=CLF_EMBED_MODEL_ID
)
llamaindex_embedding = LangchainEmbedding(embeddings)

quizz_chain = LLMChain(prompt=QUIZ_GEN_PROMPT, llm=langchain_llm)


def connect_to_weaviate():
    return weaviate.Client(
        url="https://streamlit-llm-hackathon-55r71kmd.weaviate.network",
        auth_client_secret=auth_config
    )


def read_sample_quizz_questions():
    with open('sample_questions.json', 'r') as json_file:
        return json.load(json_file)


def remove_local_dir(local_dir_path):
    print(f'Removing local files in {local_dir_path}')
    shutil.rmtree(local_dir_path, ignore_errors=True)


def load_audio():
    loader = YoutubeAudioLoader([youtube_link], st.session_state.save_dir)
    list(loader.yield_blobs())


def log_to_langsmith():
    os.environ['LANGCHAIN_API_KEY'] = st.secrets['LANGCHAIN_API_KEY']
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_PROJECT"] = PROJECT_NAME


def check_video_url():
    checker_url = f"https://www.youtube.com/oembed?url={youtube_link}"
    response = requests.get(checker_url)
    return response.status_code == 200


def extract_youtube_video_id():
    # Regular expression to match YouTube video IDs
    pattern = r"((?<=(v|V)/)|(?<=be/)|(?<=(\?|\&)v=)|(?<=embed/))([\w-]+)"
    return match.group() if (match := re.search(pattern, youtube_link)) else None


def get_video_name():
    checker_url = f"https://www.youtube.com/oembed?url={youtube_link}"
    return requests.get(checker_url).json()['title']


def index_yt_transcript(video_text):
    transcript_doc = Document(text=video_text)
    transcript_doc.metadata['title'] = vid_name
    transcript_doc.metadata['type'] = 'YT video'
    index_in_weaviate([transcript_doc])


def index_pdf():
    loader = OnlinePDFLoader(pdf_url)
    lch_docs = loader.load()
    docs = [Document.from_langchain_format(doc) for doc in lch_docs]
    for doc in docs:
        doc.metadata['title'] = pdf_url
        doc.metadata['type'] = 'pdf'
    index_in_weaviate(docs)


def get_docs_title_and_type():
    client = connect_to_weaviate()
    res = client.query.get(st.session_state.weaviate_class_name,
                           ['title', 'type']).do()
    return pd.DataFrame(res['data']['Get'][st.session_state.weaviate_class_name]).drop_duplicates()


def extract_nodes_from_documents(docs: list[Document]):
    metadata_extractor = MetadataExtractor(
        extractors=[
            QuestionsAnsweredExtractor(questions=1, llm=qa_llm, prompt_template=QA_GEN_PROMPT),
        ],
    )
    node_parser = SimpleNodeParser.from_defaults(
        chunk_size=500, chunk_overlap=100,
        metadata_extractor=metadata_extractor,
    )
    try:
        return node_parser.get_nodes_from_documents(docs)
    except Exception:
        st.error("A timeout has occurred on Clarifai's side when indexing your document. "
                 "Please try again later.")
        st.stop()


def index_in_weaviate(docs: list[Document]):
    create_schema()
    service_context = ServiceContext.from_defaults(embed_model=llamaindex_embedding,
                                                   llm=llamaindex_llm
                                                   )
    extracted_nodes = extract_nodes_from_documents(docs)
    client = connect_to_weaviate()
    weaviate_vector_store = WeaviateVectorStore(weaviate_client=client,
                                                index_name=st.session_state.weaviate_class_name)
    weaviate_index = VectorStoreIndex.from_vector_store(vector_store=weaviate_vector_store,
                                                        service_context=service_context,
                                                        )
    weaviate_index.insert_nodes(extracted_nodes)


def create_schema():
    class_schema = {
        "class": st.session_state.weaviate_class_name,
        "vectorizer": "none",
        "description": "Documents uploaded by user",
        "vectorIndexConfig": {
            "distance": "cosine",
        },
        "properties": [
            {
                "name": "text",
                "dataType": ["text"],
            },
            {
                "name": "questions_this_excerpt_can_answer",
                "dataType": ["text"],
            },
            {
                "name": "title",
                "dataType": ["text"]
            },
            {
                "name": "type",
                "dataType": ["text"]
            }
        ]
    }
    client = connect_to_weaviate()
    class_exists = client.schema.exists(st.session_state.weaviate_class_name)
    if not class_exists:
        client.schema.create_class(class_schema)


def extract_video_text():
    st.session_state.save_dir = f'vids/{youtube_video_id}'
    try:
        return transcribe(youtube_video_id)
    except Exception as e:
        st.error(e)
    finally:
        if 'save_dir' in st.session_state:
            remove_local_dir(st.session_state.save_dir)


def fetch_context_question_from_weaviate(topic: str | None,
                                         num_of_questions_to_generate: int) -> list[dict]:
    client = connect_to_weaviate()
    query = (client.query
             .get(st.session_state.weaviate_class_name,
                  ["text", "questions_this_excerpt_can_answer"]))
    if topic:
        near_vector = {
            "vector": embeddings.embed_query(topic)
        }
        result = query.with_near_vector(near_vector).with_autocut(1).with_limit(num_of_questions_to_generate).do()
        return result["data"]["Get"][st.session_state.weaviate_class_name]
    else:
        query_result = query.with_limit(num_of_questions_to_generate + 5).do()
        result = query_result["data"]["Get"][st.session_state.weaviate_class_name]
        return random.sample(result, k=min(num_of_questions_to_generate, len(result)))


async def async_generate_question(context_question_obj):
    context = context_question_obj["text"]
    question = context_question_obj["questions_this_excerpt_can_answer"]
    result = await quizz_chain.arun(context=context, question=question)
    return json.loads(result)


async def generate_quiz(context_question_list: list[dict]):
    tasks = [async_generate_question(obj) for obj in context_question_list]
    return await asyncio.gather(*tasks)


def get_quiz(context_question_list: list[dict]):
    st.session_state.quiz_questions = asyncio.run(generate_quiz(context_question_list))


@st.cache_data(show_spinner="Transcribing the video")
def transcribe(youtube_video_id):
    load_audio()
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(
        f'{st.session_state.save_dir}/{os.listdir(st.session_state.save_dir)[0]}'
    )
    return transcript.text


log_to_langsmith()

st.set_page_config(
    page_title="Quiz-Wiz",
    page_icon=":brain:",
    layout="wide",
    initial_sidebar_state="expanded",
)

if 'weaviate_class_name' not in st.session_state:
    st.session_state.weaviate_class_name = f'{WEAVIATE_CLASS_PREFIX}_{uuid.uuid4().hex}'

with st.sidebar:
    with open('app_description.md') as descr:
        st.write(descr.read())

warning_col, reset_col = st.columns(2)
with warning_col:
    st.warning(":heavy_exclamation_mark: **Note**: Processing takes quite a while.  \n"
               ":pray: Please be patient, the result will eventually arrive!")
with reset_col:
    if st.button('Reset',
                 type='primary',
                 help='Remove all the index documents and start from scratch'):
        client = connect_to_weaviate()
        client.schema.delete_class(st.session_state.weaviate_class_name)
        if 'indexed' in st.session_state:
            st.session_state.pop('indexed')
        if 'quiz_questions' in st.session_state:
            st.session_state.pop('quiz_questions')
        st.experimental_rerun()

select_sources_col, generate_questions_col, quizz_col = st.columns([1, 1, 2])

with select_sources_col:
    st.header("Documents upload")
    sources_form = st.form('sources')
    with sources_form:
        youtube_link = st.text_input('Link to YT video',
                                     value='https://www.youtube.com/watch?v=Kf3LeaUGwlg')

        video_placeholder = st.empty()

        pdf_url = st.text_input('Add PDF URL')

        index_sources = st.form_submit_button('Add sources',
                                              help='Add sources to be quizzed about')
        if youtube_link and not check_video_url():
            st.warning('Please input a valid Youtube video link')
            st.stop()
        if youtube_link:
            video_placeholder.video(youtube_link)

        if pdf_url and '.pdf' not in pdf_url:
            st.warning('Please choose a valid pdf')
            pdf_url = None

if index_sources:
    if youtube_link:
        vid_name = get_video_name()
        youtube_video_id = extract_youtube_video_id()
        with st.spinner("Indexing video"):
            index_yt_transcript(video_text=extract_video_text())
        st.session_state['indexed'] = True

    if pdf_url:
        with st.spinner("Indexing pdf"):
            index_pdf()
        st.session_state['indexed'] = True

generate_button_disabled = 'indexed' not in st.session_state

with generate_questions_col:
    st.header("Generate the quiz")
    if not generate_button_disabled:
        st.subheader("Available sources")
        st.dataframe(get_docs_title_and_type(),
                     hide_index=True,
                     use_container_width=True)
    with st.form('generate'):
        num_of_questions_to_generate = st.number_input('Max number of question to generate',
                                                       min_value=1,
                                                       max_value=5,
                                                       value=3,
                                                       disabled=generate_button_disabled)

        freetext_topic = st.text_input("Enter a topic that you're interested in (Optional):",
                                       disabled=generate_button_disabled)
        generate_quiz_button = st.form_submit_button('Generate questions',
                                                     disabled=generate_button_disabled)
if generate_quiz_button:
    with st.spinner("Generating quiz"):
        context_question = fetch_context_question_from_weaviate(freetext_topic, num_of_questions_to_generate)
        try:
            get_quiz(context_question)
        except Exception as e:
            st.error("A timeout has occurred on Clarifai's side when generating the quiz questions. "
                     "Please try again later")
            st.stop()

if 'quiz_questions' in st.session_state:
    with quizz_col:
        st.header("Test your knowledge")
        with st.form('quiz'):
            # quiz_questions = read_sample_quizz_questions()
            quiz_questions = st.session_state.quiz_questions
            number_of_questions = len(quiz_questions)
            if number_of_questions > 1:
                pass_score = st.select_slider('Pass score',
                                              range(1, number_of_questions + 1),
                                              value=math.ceil(0.8 * number_of_questions))
            else:
                pass_score = 1
            answers = []
            containers = []
            for i, question in enumerate(quiz_questions):
                container = st.container()
                answer = container.radio(f'Question {i + 1}: {question["question"]}',
                                         question['answers'],
                                         )
                answers.append(question['answers'].index(answer))
                containers.append(container)
            submit_quiz = st.form_submit_button('Submit my answers')

        if submit_quiz:
            score = 0
            for question, answer_index, container in zip(quiz_questions, answers, containers):
                rationale = question['rationales'][answer_index]
                if answer_index == question['correct']:
                    container.success(rationale)
                    score += 1
                else:
                    container.error(rationale)

            message = f'Your final score is: {score}/{number_of_questions}'
            if score >= pass_score:
                st.success(message)
                st.success(':partying_face: Well done! Keep it up!')
            else:
                st.error(message)
                st.error('Not this time :grimacing: Please Try again!')
