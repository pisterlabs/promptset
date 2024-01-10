import pandas as pd
import streamlit as st
from langchain.llms.openai import OpenAI
from llama_index import Document, VectorStoreIndex, get_response_synthesizer
from llama_index.indices.postprocessor import (SentenceEmbeddingOptimizer,
                                               SimilarityPostprocessor)


def extract_json_data_to_index(json_file):
    if json_file is not None:
        # Can be used wherever a "file-like" object is accepted:
        dataframe = pd.read_json(json_file)
        st.write("Data preview:")
        st.write(dataframe)

        # Extract the specific data
        result_dict = {
                "Synthetic user": "",
                "Problems": "",
                "Solution": "",
                "User Interviews": []
            }
        for i in range(len(dataframe)):
            data = dataframe.iloc[i]
            result_dict["Synthetic user"] = data["Synthetic user"]
            result_dict["Problems"] = data["Problems"]
            result_dict["Solution"] = data["Solution"]
            result_dict["User Interviews"].append("\n".join([data[f"Question {j}"] for j in range(1, 11)]))
        
        return result_dict
    
def index_user_interviews(data) -> VectorStoreIndex:
    documents = [Document(text=user_interview, metadata={"type": "user_interview"}) for user_interview in data["User Interviews"]]
    index = VectorStoreIndex.from_documents(documents=documents)
    return index