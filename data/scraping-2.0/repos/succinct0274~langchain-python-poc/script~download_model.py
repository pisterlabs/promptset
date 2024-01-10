from transformers.pipelines import pipeline
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.llms.huggingface_pipeline import HuggingFacePipeline

hf_embedding = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        encode_kwargs={'normalize_embeddings': False}
)

summarization_model_name = 'pszemraj/led-large-book-summary'
summarization_llm = HuggingFacePipeline(pipeline=pipeline('summarization', summarization_model_name, max_length=300))