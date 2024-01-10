import os

from dotenv import load_dotenv
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import DeepLake, Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

from utils.fetchRepos import getRepo


#from src.utils.fetchRepos import getTestRepo,readFromExcel

def indexRepo(repoURL):
    load_dotenv()

    os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

    #Set to true if you want to include documentation files
    documentation = True

    embeddings = OpenAIEmbeddings(disallowed_special=())
    repoDir = getRepo(repoURL)

    #load files in repo
    root_dir = repoDir
    #root_dir = getTestRepo()
    print("project directory is: "+root_dir)
    #get name of repo
    repo_name = root_dir.split("/")[-1]
    print("repo name is: "+repo_name)

    #check if repo is already indexed
    if os.path.exists("vectordbs/"+repo_name):
        print("repo already indexed")
        return str("vectordbs/"+repo_name)

    fileextensions = [
    ".ts", ".json", ".js", ".jsx", ".tsx", ".html", ".css", ".scss", ".less", ".py", ".java", ".cpp", ".h", ".c",
    ".cs", ".go", ".php", ".rb", ".swift", ".kt", ".dart", ".rs", ".sh", ".yml", ".yaml", ".xml", ".txt"]

    if documentation:
        fileextensions.append("README.md")
        repo_name = repo_name + "_doc"
        print("added documentation files to index")

    docs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            #ignore node_modules and package-lock.json
            if "node_modules" in dirpath or "package-lock.json" in file:
                continue
            #ignore files that are not of the specified file extensions
            if file.endswith(tuple(fileextensions)):
                try:
                    loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
                    docs.extend(loader.load_and_split())
                    for doc in docs:
                        doc.metadata['file_name'] = file
                except Exception as e:
                    pass

    #chunk the files
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    texts = text_splitter.split_documents(docs)

    #embed the files and add them to the vector db
    db = DeepLake(dataset_path="vectordbs/"+repo_name, embedding_function=embeddings) #dataset would be publicly available
    db.add_documents(texts)

    return str("vectordbs/"+repo_name)