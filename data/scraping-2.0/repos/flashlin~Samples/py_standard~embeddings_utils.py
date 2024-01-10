from langchain.embeddings import HuggingFaceInstructEmbeddings


def load_huggingface_instructor_embeddings():
    instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl",
                                                          model_kwargs={"device": "cuda"})
    return instructor_embeddings
