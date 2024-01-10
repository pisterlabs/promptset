import openai
import os
import pathspec
import subprocess
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from utils.faiss_utils import create_store

# Set the OpenAI API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]


def clone_or_pull_repository(repo_url, local_path):
    # if the repository is already cloned, pull the latest changes
    if os.path.isdir(local_path):
        subprocess.run(["git", "pull"], cwd=local_path)
        return

    """Clone the specified git repository to the given local path."""
    subprocess.run(["git", "clone", repo_url, local_path])


def debug_print_docs(docs):
    """Print the input documents with a divider between them."""
    for doc in docs:
        print(doc)
        print("=" * 80)


def get_language_enum(s: str) -> Language:
    special_cases = {
        "jsx": "js",
        "ts": "js",
        "tsx": "js",
        "mjs": "js",
        "svelte": "js",
        "astro": "js",
    }

    if s in special_cases:
        s = special_cases[s]

    for e in Language.__members__.values():
        if s == e.value:
            return e

    return None


def dynamic_load_and_split_docs(file_path):
    """
    Load documents from the specified file path.
    Uses a different document loader based on the extension of the document.
    """
    docs = None

    file_extension = file_path.split(".")[-1]
    file_extension = file_extension.lower()
    print(f"PATH: {file_path}, EXT: {file_extension}")

    language = get_language_enum(file_extension)
    # if language is not None:
    #     print(f"LANG: {language}")

    if file_extension == "md" or file_extension == "mdx":
        """Markdown"""
        text = open(file_path, "r").read()
        markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on
        )
        docs = markdown_splitter.split_text(text)
        print(f"Split markdown into {len(docs)} chunks")
    elif language:
        """Code"""
        code_splitter = RecursiveCharacterTextSplitter.from_language(
            language=language, chunk_size=2000, chunk_overlap=600
        )
        code = open(file_path, "r").read()
        docs = code_splitter.create_documents([code])
        print(f"Split {language} code  into {len(docs)} chunks")
    elif file_extension == "pdf":
        """PDF"""
        loader = PyPDFLoader(file_path)
        docs = loader.load_and_split()
        print(f"Split PDF into {len(docs)} chunks")
    elif file_extension == "csv":
        """CSV"""
        loader = CSVLoader(file_path)
        data = loader.load()
        csv_splitter = CharacterTextSplitter(chunk_size=8000, chunk_overlap=0)
        docs = csv_splitter.split_documents(data)
        print(f"Split CSV into {len(docs)} chunks")
    # elif file_extension == "json":
    #     """JSON - not working"""
    #     data = json.loads(Path(file_path).read_text())
    #     splitter = CharacterTextSplitter(chunk_size=80, chunk_overlap=0)
    #     docs = splitter.create_documents(data)
    #     print(f"Split JSON into {len(docs)} chunks")
    else:
        """Text"""
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load_and_split()
        print(f"Split text into {len(docs)} chunks")

    # debug_print_docs(docs)
    return docs


def load_docs(root_dir, file_extensions=None):
    """
    Load documents from the specified root directory.
    Ignore dotfiles, dot directories, and files that match .gitignore rules.
    Optionally filter by file extensions.
    """

    docs = []

    # Load .gitignore rules
    gitignore_path = os.path.join(root_dir, ".gitignore")

    if os.path.isfile(gitignore_path):
        with open(gitignore_path, "r") as gitignore_file:
            gitignore = gitignore_file.read()
        spec = pathspec.PathSpec.from_lines(
            pathspec.patterns.GitWildMatchPattern, gitignore.splitlines()
        )
    else:
        spec = None

    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Remove dot directories from the list of directory names
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]

        for file in filenames:
            file_path = os.path.join(dirpath, file)

            # Skip dotfiles
            if file.startswith("."):
                continue

            # Skip files that match .gitignore rules
            if spec and spec.match_file(file_path):
                continue

            if file_extensions and os.path.splitext(file)[1] not in file_extensions:
                continue

            try:
                new_docs = dynamic_load_and_split_docs(file_path)
                docs.extend(new_docs)
            except Exception:
                pass

    return docs


def ingest(repo_url, include_file_extensions):
    """
    Ingest a git repository by cloning it, filtering files, splitting documents,
    creating embeddings, and storing everything in a FAISS index.
    """

    repo_name = repo_url.split("/")[-1].replace(".git", "")

    local_path = f"repos/{repo_name}"

    clone_or_pull_repository(repo_url, local_path)

    docs = load_docs(local_path, include_file_extensions)
    print(f"Loaded {len(docs)} documents")

    create_store(repo_name, docs)
