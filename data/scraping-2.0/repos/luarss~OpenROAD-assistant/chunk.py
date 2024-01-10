from langchain.document_loaders import GitLoader
from langchain.schema import Document
import argparse
from utils import save_docs_to_jsonl, load_docs_from_jsonl
from chunk_strategies import text_splitter

parser = argparse.ArgumentParser()
parser.add_argument("--exts", type=str, nargs="+", help="List of file extensions to process (e.g., md, cpp, cc)",
                        default=[".md"]) 
parser.add_argument("--chunk-size", type=int, help="Chunk size for text splitter")
parser.add_argument("--overlap", type=int, help="Overlap for text splitter")
args = parser.parse_args()
exts = args.exts
chunk_size = args.chunk_size
overlap = args.overlap

def main(chunk_size, overlap):
    loader = GitLoader(
        clone_url="https://github.com/The-OpenROAD-Project/OpenROAD",
        repo_path="./data",
        branch="master",
        file_filter = lambda file_path: any(file_path.endswith(ext) for ext in exts)
    )
    data = loader.load()
    final = text_splitter(data, chunk_size, overlap)
    save_docs_to_jsonl(final,'./tempdata/data.jsonl')
    final2=load_docs_from_jsonl('./tempdata/data.jsonl')
    assert(len(final2) == len(final))
    print(f'Chunking done, final chunk len {len(final)}')

if __name__ == "__main__":
    main(chunk_size, overlap)