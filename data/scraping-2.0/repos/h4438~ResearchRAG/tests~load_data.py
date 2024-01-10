from langchain.text_splitter import RecursiveCharacterTextSplitter
from collections import Counter


def load_splitted_major_docs(path):
    docs = pd.read_csv(f"{path}")
    docs = docs[docs['id'] >= 6]

    db_df = docs
    db = []
    dist = []

    text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n\n","\n\n", "\n"],
    chunk_size=1800,
    chunk_overlap=120,
    length_function=len,
    is_separator_regex=False
    )

    for i, data in db_df.iterrows():
        doc = text_splitter.create_documents([data['content']])
        dist.append(len(doc))
        [d.metadata.update({"id": data['id']}) for d in doc]
        db.extend(doc)
    print(Counter(dist))
    print(len(db))
    return db
