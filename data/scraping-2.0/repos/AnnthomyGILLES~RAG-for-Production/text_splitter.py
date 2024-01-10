from langchain.text_splitter import RecursiveCharacterTextSplitter
from ray.data import from_items


def chunk_section(section, chunk_size=300, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.create_documents(
        texts=[section["text"]], metadatas=[{"source": section["source"]}]
    )
    return [
        {"text": chunk.page_content, "source": chunk.metadata["source"]}
        for chunk in chunks
    ]


if __name__ == "__main__":
    # Sample data for sections_ds
    sample_sections = [
        {
            "text": "Section 1 text goes here. More details about section 1.",
            "source": "Source 1",
        },
        {
            "text": "Section 2 has different content. It might be longer or shorter.",
            "source": "Source 2",
        },
        {
            "text": "Another section, Section 3, with its own unique text and information.",
            "source": "Source 3",
        },
    ]

    # Convert list to a Ray dataset
    sections_ds = from_items(sample_sections)
    # Define chunking parameters
    chunk_size = 300
    chunk_overlap = 50
    separators = ["\n\n", "\n", " ", ""]

    chunks_ds = sections_ds.flat_map(chunk_section)
    print(f"{chunks_ds.count()} chunks")
    chunks_ds.show(1)
