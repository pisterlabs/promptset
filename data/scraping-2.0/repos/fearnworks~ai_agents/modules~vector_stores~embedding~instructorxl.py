from langchain.embeddings import HuggingFaceInstructEmbeddings


def get_default_instructor_embedding():
    instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl", 
                                                        model_kwargs={"device": "cuda"})
    return instructor_embeddings