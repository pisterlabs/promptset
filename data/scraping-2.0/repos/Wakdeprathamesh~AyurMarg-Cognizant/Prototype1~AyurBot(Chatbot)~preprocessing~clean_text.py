# import re


# def clean_text(filename):
#   with open(filename, 'rb') as file:
#         text = file.read().decode('utf-8')

#   pattern = r'[^\w\s]'

#   cleaned_text = re.sub(pattern, '', text)

#   with open(filename, 'w') as f:
#     f.write(cleaned_text)

# clean_text("D:\\Computer_Programming\\Python\\Hackathon2\\datasets\\astanga-hridaya-sutrasthan-handbook.txt")

# import os
# import pickle
# from langchain.vectorstores import FAISS
# from langchain.embeddings.huggingface import HuggingFaceInstructEmbeddings
# import streamlit as st

# def get_vector(text, embeddings_cache_path):
#     if os.path.exists(embeddings_cache_path):
#         st.info('Embeddings retrieved!')
#         with open(embeddings_cache_path, 'rb') as file:
#             serialized_faiss = pickle.load(file)
#             embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
#             vectorstore = FAISS.deserialize_from_bytes(embeddings=embeddings, serialized=serialized_faiss)
#     else:
#         st.info("Computing embeddings....")
#         embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
#         vectorstore = FAISS.from_texts(texts=text, embedding=embeddings)
        
#         # Serialize only the FAISS index
#         serialized_faiss = vectorstore.serialize_to_bytes()

#         # Save the serialized FAISS index
#         with open(embeddings_cache_path, 'wb') as file:
#             pickle.dump(serialized_faiss, file)

#         st.success("Embeddings computation completed and saved to cache.")
#     return vectorstore

# # Example usage
# texts = ["example text 1", "example text 2"]
# embeddings_cache_path = "embeddings_cache.pkl"
# vectorstore = get_vector(texts, embeddings_cache_path)


# import multiprocessing
# import numpy as np

# def stress_cpu_core(core_id, duration_seconds=10):
#     # Use NumPy for computations to stress the CPU
#     start_time = np.datetime64('now')
#     while (np.datetime64('now') - start_time).astype(int) / 1e9 < duration_seconds:
#         np.linalg.eig(np.random.rand(1000, 1000))  # Example NumPy operation

# if __name__ == "__main__":
#     # Get the number of available CPU cores
#     num_cores = multiprocessing.cpu_count()

#     # Create a process pool to stress each core
#     with multiprocessing.Pool(processes=num_cores) as pool:
#         # Run stress_cpu_core on each core in parallel
#         pool.starmap(stress_cpu_core, [(core_id,) for core_id in range(num_cores)])

