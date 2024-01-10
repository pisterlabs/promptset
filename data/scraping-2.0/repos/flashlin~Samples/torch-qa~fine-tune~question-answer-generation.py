from langchain.text_splitter import TokenTextSplitter, TextSplitter, Tokenizer, split_text_on_tokens
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from transformers import AutoTokenizer
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceBgeEmbeddings
from torch import bfloat16
import transformers


MODEL = "../models/neural-chat-7b-v3-16k.Q4_K_M.gguf"
EMBEDDING_MODEL = "../models/bge-base-en"


def create_local_text_splitter(model_name, chunk_size, chunk_overlap):
    tokenizer = AutoTokenizer.from_pretrained("../models/neural-chat-7b-v3-16k")

    def local_tokenizer_encode(text: str) -> list[int]:
        return tokenizer.encode(text)

    def local_tokenizer_decode(token_ids: list[int]) -> str:
        return tokenizer.decode(token_ids)

    tokenizer = Tokenizer(
        chunk_overlap=chunk_overlap,
        tokens_per_chunk=chunk_size,
        decode=local_tokenizer_decode,
        encode=local_tokenizer_encode,
    )
    # text_splitter = TextSplitter.from_huggingface_tokenizer(tokenizer,
    #                                                         chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return tokenizer


def create_openai_text_splitter(chunk_size, chunk_overlap):
    splitter = TokenTextSplitter(
        model_name='gpt-3.5-turbo',
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter


def load_model(model_id, device: str = "auto"):
    print(f"loading model {model_id}")
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )
    model_config = transformers.AutoConfig.from_pretrained(
        model_id,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=False,
        config=model_config,
        quantization_config=bnb_config,
        device_map=device,
        load_in_8bit=True,
    )
    return model


def create_llm_model(model, temperature):
    # llm_pipeline = ChatOpenAI(
    #     temperature=0.3,
    #     model="gpt-3.5-turbo"
    # )
    llm_pipeline = load_model(MODEL)
    return llm_pipeline


def create_llm_embedding(model_name):
    #return OpenAIEmbeddings()
    print(f"loading embeddings {model_name}")
    encode_kwargs = {'normalize_embeddings': True}  # set True to compute cosine similarity
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cuda'},
        encode_kwargs=encode_kwargs
    )
    return embeddings


def pdf_file_processing(file_path):
    loader = PyPDFLoader(file_path)
    data = loader.load()

    question_gen = ''
    for page in data:
        question_gen += page.page_content

    splitter_ques_gen = create_local_text_splitter(MODEL, chunk_size=10000, chunk_overlap=200)

    chunks_ques_gen = splitter_ques_gen.split_text(question_gen)

    document_ques_gen = [Document(page_content=t) for t in chunks_ques_gen]

    splitter_ans_gen = create_local_text_splitter(model_name=MODEL,
                                                  chunk_size=1000, chunk_overlap=100)
    document_answer_gen = splitter_ans_gen.split_documents(document_ques_gen)

    return document_ques_gen, document_answer_gen


def llm_pipeline(file_path):
    document_ques_gen, document_answer_gen = pdf_file_processing(file_path)

    llm_ques_gen_pipeline = create_llm_model(model="gpt-3.5-turbo",
                                             temperature=0.3)

    prompt_template = """
    You are an expert at creating questions based on coding materials and documentation.
    Your goal is to prepare a coder or programmer for their exam and coding tests.
    You do this by asking questions about the text below:

    ------------
    {text}
    ------------

    Create questions that will prepare the coders or programmers for their tests.
    Make sure not to lose any important information.

    QUESTIONS:
    """

    PROMPT_QUESTIONS = PromptTemplate(template=prompt_template, input_variables=["text"])

    refine_template = ("""
    You are an expert at creating practice questions based on coding material and documentation.
    Your goal is to help a coder or programmer prepare for a coding test.
    We have received some practice questions to a certain extent: {existing_answer}.
    We have the option to refine the existing questions or add new ones.
    (only if necessary) with some more context below.
    ------------
    {text}
    ------------

    Given the new context, refine the original questions in English.
    If the context is not helpful, please provide the original questions.
    QUESTIONS:
    """)

    REFINE_PROMPT_QUESTIONS = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
    )

    ques_gen_chain = load_summarize_chain(llm=llm_ques_gen_pipeline,
                                          chain_type="refine",
                                          verbose=True,
                                          question_prompt=PROMPT_QUESTIONS,
                                          refine_prompt=REFINE_PROMPT_QUESTIONS)

    ques = ques_gen_chain.run(document_ques_gen)

    embeddings = create_llm_embedding()

    vector_store = FAISS.from_documents(document_answer_gen, embeddings)

    llm_answer_gen = create_llm_model(model="gpt-3.5-turbo", temperature=0.1)

    ques_list = ques.split("\n")
    filtered_ques_list = [element for element in ques_list if element.endswith('?') or element.endswith('.')]

    answer_generation_chain = RetrievalQA.from_chain_type(llm=llm_answer_gen,
                                                          chain_type="stuff",
                                                          retriever=vector_store.as_retriever())

    return answer_generation_chain, filtered_ques_list


answer_generation_chain, ques_list = llm_pipeline('./data/ai.pdf')

for question in ques_list:
    print("Question: ", question)
    answer = answer_generation_chain.run(question)
    print("Answer: ", answer)
    print("--------------------------------------------------\n\n")
