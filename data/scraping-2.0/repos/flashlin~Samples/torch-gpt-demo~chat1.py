import torch
import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, pipeline
from langchain.chains import RetrievalQA
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from embeddings_utils import load_huggingface_instructor_embeddings
from pdf_utils import load_and_split_pdf_texts_from_directory
from vectordb_utils import load_chroma_from_documents
import textwrap


# conda install -c conda-forge transformers
model_name = "TheBloke/wizardLM-7B-HF"        # 需要 16GB VRAM
model_name = "TheBloke/stable-vicuna-13B-HF"  # 26GB VRAM
model_name = 'Tribbiani/vicuna-7b'
tokenizer = LlamaTokenizer.from_pretrained(model_name)

model = LlamaForCausalLM.from_pretrained(model_name,
                                         load_in_8bit=True,
                                         device_map='auto',
                                         torch_dtype=torch.float16,
                                         low_cpu_mem_usage=True,
                                         )

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=1024,
    temperature=0,
    top_p=0.95,
    repetition_penalty=1.15
)
local_llm = HuggingFacePipeline(pipeline=pipe)

texts = load_and_split_pdf_texts_from_directory('./documents')
instructor_embeddings = load_huggingface_instructor_embeddings()

vectordb = load_chroma_from_documents(texts, instructor_embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

qa_chain = RetrievalQA.from_chain_type(llm=local_llm,
                                       chain_type="stuff",
                                       retriever=retriever,
                                       return_source_documents=True)


def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text


def process_llm_response(llm_response):
    print(wrap_text_preserve_newlines(llm_response['result']))
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])


def query_question(query):
    llm_response = qa_chain(query)
    process_llm_response(llm_response)


query_question("What is Flash attention?")
