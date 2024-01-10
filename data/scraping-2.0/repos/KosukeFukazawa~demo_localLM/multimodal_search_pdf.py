import base64
import io
from pathlib import Path
from PIL import Image

from unstructured.partition.pdf import partition_pdf
from langchain.vectorstores.chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from japanese_clip import JapaneseCLIPEmbeddings

def img_from_base64(base64_str):
    img_data = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_data))
    return img

def is_base64(s):
    """Check if a string is Base64 encoded"""
    try:
        return base64.b64encode(base64.b64decode(s)) == s.encode()
    except Exception:
        return False

data_dir = Path("data/paper_similarity")
pdf_dir = data_dir / "pdf"
pdf_name = "gart_paper_arxiv2023.pdf"
image_dir = data_dir / "images"
image_dir.mkdir(parents=True, exist_ok=True)
text_dir = data_dir / "texts"
text_dir.mkdir(parents=True, exist_ok=True)
chroma_dir = data_dir / "chroma_db"
chroma_exists = chroma_dir.exists()

output_dir = data_dir / "output"
output_dir.mkdir(exist_ok=True)

embeddings = JapaneseCLIPEmbeddings(device_map="mps") 
vectorstore = Chroma(
    collection_name="mm_retrieve_clip_photos", 
    embedding_function=embeddings,
    persist_directory=str(chroma_dir)
)
if not chroma_exists:
    # Extract images, tables, and chunk text
    raw_pdf_elements = partition_pdf(
        filename= str(pdf_dir / pdf_name),
        extract_images_in_pdf=True,
        infer_table_structure=True,
        chunking_strategy="by_title",
        strategy="hi_res",
        image_output_dir_path=str(image_dir),
    )

    # Categorize text elements by type
    tables = []
    texts = []
    with open(str(text_dir / "text.txt"), "w") as f, \
         open(str(text_dir / "table.txt"), "w") as g:
        for element in raw_pdf_elements:
            if "unstructured.documents.elements.Table" in str(type(element)):
                tables.append(str(element))
                g.write(str(element))
                g.write("\n")
            elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
                texts.append(str(element))
                f.write(str(element))
                f.write("\n")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, chunk_overlap=20,
    )
    
    # Add images and texts to vectorstore
    image_uris = list(
        map(lambda x: str(x), 
            set(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")))
        )
    )
    vectorstore.add_images(image_uris)
    vectorstore.add_texts(texts)

test_query = input("Enter a query: ")
docs_and_scores = vectorstore.similarity_search_with_score(test_query, k=10)
with open(output_dir / "query.txt", "w") as f:
    f.write(test_query)
for i, (doc, score) in enumerate(docs_and_scores):
    print(f"Rank {i+1:02d}: (score: {score:.4f})")
    doc = doc.page_content
    if is_base64(doc):
        img = img_from_base64(doc)
        img.save(output_dir / f"{i:03d}.jpg")
    else:
        with open(output_dir / f"{i:03d}.txt", "w") as f:
            f.write(doc)
