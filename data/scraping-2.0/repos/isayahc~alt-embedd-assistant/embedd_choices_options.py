import streamlit as st
from typing import List

from llama_index.vector_stores import MilvusVectorStore

from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInferenceAPIEmbeddings

from llama_index.storage.storage_context import StorageContext
from generate_wiki_docs import wiki_docs

from llama_index import (
    VectorStoreIndex,
    # SimpleWebPageReader,
    LLMPredictor,
    ServiceContext
)

from langchain.llms import VertexAI

from trulens_eval.feedback import Groundedness


from trulens_eval import TruLlama, Feedback, Tru, feedback

from dotenv import load_dotenv
import os

import numpy as np
    
load_dotenv()
tru = Tru()



CLUSTER_ENDPOINT = os.getenv("ZILLZ_CLUSTER_ENDPOINT")
TOKEN = os.getenv("ZILLIZ_TOKEN")
COLLECTION_NAME = os.getenv("ZILLIZ_COLLECTION_NAME")
GOOGLE_PROJECT_NAME = os.getenv("GOOGLE_PROJECT_NAME")
PRIVATE_EMBEDDING_MODEL_ENDPOINT = os.getenv("PRIVATE_EMBEDDING_MODEL_ENDPOINT")

llm = VertexAI(project=GOOGLE_PROJECT_NAME)

# embed_v12 = HuggingFaceInferenceAPIEmbeddings(
#         model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
#         api_key = os.getenv("HUGGINGFACE_API_KEY"),
#         api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
#     )


embed_ft3_v12 = HuggingFaceInferenceAPIEmbeddings(
        model_name = "Sprylab/paraphrase-multilingual-MiniLM-L12-v2-fine-tuned-3",
        api_key = os.getenv("HUGGINGFACE_API_KEY"),
        api_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/paraphrase-multilingual-MiniLM-L12-v2-fine-tuned-3"
    )



bge_large_en = HuggingFaceInferenceAPIEmbeddings(
        model_name = "BAAI/bge-large-en-v1.5",
        api_key = os.getenv("HUGGINGFACE_API_KEY"),
        api_url = "https://api-inference.huggingface.co/models/BAAI/bge-large-en-v1.5"
    )

personal_bge_large_en = HuggingFaceInferenceAPIEmbeddings(
        model_name = "sentence-transformers/all-MiniLM-L6-v2",
        api_key = os.getenv("HUGGINGFACE_API_KEY"),
        api_url = PRIVATE_EMBEDDING_MODEL_ENDPOINT
    )


# Initialize OpenAI-based feedback function collection class:
openai_gpt35 = feedback.OpenAI(model_engine="gpt-3.5-turbo")

# Define groundedness
grounded = Groundedness(groundedness_provider=openai_gpt35)
f_groundedness = Feedback(grounded.groundedness_measure_with_cot_reasons, name = "Groundedness").on(
    TruLlama.select_source_nodes().node.text.collect() # context
).on_output().aggregate(grounded.grounded_statements_aggregator)

# Question/answer relevance between overall question and answer.
f_qa_relevance = Feedback(openai_gpt35.relevance_with_cot_reasons, name = "Answer Relevance").on_input_output()

# Question/statement relevance between question and each context chunk.
f_qs_relevance = Feedback(openai_gpt35.qs_relevance_with_cot_reasons, name = "Context Relevance").on_input().on(
    TruLlama.select_source_nodes().node.text
).aggregate(np.max)

        
def main():
    st.subheader("Set Hugging Face Token")
    hugging_face_token = st.text_input("Enter your Hugging Face Token:", value="")
    if hugging_face_token:
        os.environ["BACKUP_HUGGING_FACE_TOKEN"] = hugging_face_token



    st.title("Streamlit UI for Option Selection")

    # Define your options
    distance_metrics = ["L2", "IP"]
    index_types = ["FLAT", "IVF_FLAT", "IVF_SQ8", "IVF_PQ", "HNSW", "IVF_HNSW", "RHNSW_FLAT", "RHNSW_SQ", "RHNSW_PQ", "ANNOY"]
    nprobe_values: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    top_k_values: List[int] = [1, 2, 3, 4, 5]

    # Form for selections
    with st.form(key='options_form'):
        selected_distance_metric = st.selectbox("Select Distance Metric", distance_metrics)
        selected_index_type = st.selectbox("Select Index Type", index_types)
        selected_nprobe_value = st.selectbox("Select Nprobe Value", nprobe_values)
        selected_chunk_overlap = st.number_input(
        "Enter Chunk Overlap",
        min_value=0,  # Minimum value
        value=30,     # Default value
        step=1,        # Increment step
        max_value=50,
    )
        selected_chunk_size = st.number_input(
        "Enter Chunk Size",
        min_value=1,  # Minimum value
        value=10,     # Default value
        step=1,        # Increment step
        max_value=50,
    )
        embedding_model_choice = st.radio(
            "Choose a model:",
            (
            # embed_v12, 
            embed_ft3_v12,
            bge_large_en, 
            personal_bge_large_en),
            format_func=lambda model: model.model_name
        )

        # Submit button
        submit_button = st.form_submit_button(label='Update Choices')

    # Displaying the selected options
    if submit_button:
        st.write(f"Selected Distance Metric: {selected_distance_metric}")
        st.write(f"Selected Index Type: {selected_index_type}")
        st.write(f"Selected Nprobe Value: {selected_nprobe_value}")

        vector_store = MilvusVectorStore(
            index_params={
                "index_type": selected_index_type,
                "metric_type": selected_distance_metric
                },
            search_params={"nprobe": selected_nprobe_value},
            # overwrite=True,
            uri=CLUSTER_ENDPOINT,
            token=TOKEN,
            collection_name=COLLECTION_NAME,
        )

        storage_context = StorageContext.from_defaults(
            vector_store = vector_store
            )

        service_context = ServiceContext.from_defaults(
            embed_model = embedding_model_choice, 
            llm = llm,
            chunk_size=selected_chunk_size,
            chunk_overlap=selected_chunk_overlap,
        )

        index = VectorStoreIndex.from_documents(
                    wiki_docs,
                    service_context=service_context,
                    storage_context=storage_context)

        query_engine = index.as_query_engine(top_k = top_k_values)

        tru_query_engine = TruLlama(query_engine,
                feedbacks=[f_groundedness, f_qa_relevance, f_qs_relevance],
                metadata={
                    'index_param':selected_index_type,
                    'embed_model':embedding_model_choice,
                    'top_k':top_k_values,
                    'chunk_size':chunk_size
                    })

        with tru_query_engine as recording:
            for prompt in test_prompts:
                tru_query_engine.query(prompt)

        data = tru.get_records_and_feedback(app_ids=[])[0]

        df = pd.DataFrame(data)
        df = df[['qs_relevance', 'relevance', 'app_id', 'type', 'input', 'output', 'ts']]

        # Display the dataframe
        st.write("Here is our sample dataframe:")
        st.dataframe(df)

        st.table(df)



if __name__ == "__main__":
    main()
