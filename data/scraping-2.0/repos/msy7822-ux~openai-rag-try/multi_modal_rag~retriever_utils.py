from langchain.schema.document import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
import uuid


def create_vectorstore(
    vectorstore,
    docstore,
    texts,
    table_summaries,
    image_summaries,
    tables,
    img_base64_list,
):
    id_key = "doc_id"
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore, docstore=docstore, id_key=id_key
    )

    # Add texts
    doc_ids = [str(uuid.uuid4()) for _ in texts]
    for i, s in enumerate(texts):
        retriever.vectorstore.add_documents(
            [Document(page_content=s, metadata={id_key: doc_ids[i]})]
        )
    retriever.docstore.mset(list(zip(doc_ids, texts)))

    # Add tables
    table_ids = [str(uuid.uuid4()) for _ in tables]
    for i, s in enumerate(table_summaries):
        retriever.vectorstore.add_documents(
            [Document(page_content=s, metadata={id_key: table_ids[i]})]
        )
    retriever.docstore.mset(list(zip(table_ids, tables)))

    # Add image summaries
    img_ids = [str(uuid.uuid4()) for _ in img_base64_list]
    for i, s in enumerate(image_summaries):
        retriever.vectorstore.add_documents(
            [Document(page_content=s, metadata={id_key: img_ids[i]})]
        )
    retriever.docstore.mset(list(zip(img_ids, img_base64_list)))

    return retriever
