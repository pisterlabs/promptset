import nest_asyncio
from llama_hub.github_repo import GithubClient, GithubRepositoryReader
import os
import utils
from utils import print_verbose
import pickle
from dotenv import load_dotenv
import openai
from llama_index import download_loader, GPTVectorStoreIndex, Document
download_loader("GithubRepositoryReader")


utils.setup()

# .pkl file where the code will store data about the respository
storeFileName = "data.pkl"


def generateDataFile(username: str, repo: str, branch: str = "main", verbose: int = 0) -> None:
    """
    A function to generate a list of document objects from a github repository. 
    Writes the list of Document objects as a .pkl in data.pkl
    """

    github_client = GithubClient(os.getenv("GITHUB_TOKEN"))

    print_verbose(
        f"INFO: {utils.getCurrentTime()} [*] Loading Github repository...", verbose)

    loader = GithubRepositoryReader(
        github_client,
        owner=username,
        repo=repo,
        verbose=False,
        concurrent_requests=10)

    data = loader.load_data(branch=branch)

    print_verbose(
        f"INFO: {utils.getCurrentTime()} [*] Storing data...", verbose)

    with open(storeFileName, "wb") as f:
        pickle.dump(data, f)


def generateQuery(query: str, verbose: int = 0) -> str:
    """
    A function to generate a query response from the given data. 
    """

    if not os.path.exists("data.pkl"):
        raise Exception("INFO: [*] Data file does not exist!")

    print_verbose(
        f"INFO: {utils.getCurrentTime()} [*] Unpacking data...", verbose)

    # Unpackage our documents object
    with open(storeFileName, "rb") as f:
        data = pickle.load(f)

    print_verbose(
        f"INFO: {utils.getCurrentTime()} [*] Generating index...", verbose)

    index = GPTVectorStoreIndex.from_documents(data)

    # Turns index into a query engine to feed questions into.
    print_verbose(
        f"INFO: {utils.getCurrentTime()} [*] Generating query engine...", verbose)

    query_engine = index.as_query_engine()

    print_verbose(
        f"INFO: {utils.getCurrentTime()} [*] Generating repsonse...", verbose)

    response = query_engine.query(query)

    return response.response


def generateFileNames(verbose: int = 0) -> list[dict]:
    """
    A function to generate file locations from our data. Returns a set of file names, starting from the github repo.

    Generate a list of dictionaries. Key: "filename", Value: filename
    """

    print_verbose(
        f"INFO: {utils.getCurrentTime()} [*] Generating file names...", verbose)

    data = getDataPKL()
    locations = []
    for document in data:
        locations.append(document.extra_info)
    return locations


def generateResponseFromFile(fileName: str) -> str:
    """
    A function to generate a detailed description for a certain file.
    This is mainly used to produce descripts for the wikipedia page.
    """

    return generateQuery(f"Write me a detailed description of the following file or class: {fileName}. The response should include a detailed list of variables and functions.")


def generateDescriptions(listOfFileNames: list[dict]) -> str:
    """
    A function that iterates through each file and and produces a description of each file.

    Description of each are appended to formed with a large string of all descriptions.

    Returns that string with all descriptions
    """

    desc = ""

    for fileNames in listOfFileNames:
        desc += generateResponseFromFile(fileNames["file_name"])

    return desc


def getDataPKL() -> list[Document]:
    """
    A function that generates a list of Document objects. Serves as data for later parsing.
    """

    # Error checking to see if our data file exists
    if not os.path.exists(storeFileName):
        raise Exception("Data file not generated!")

    with open(storeFileName, "rb") as f:
        data = pickle.load(f)
        return data


# (DEBUGGING)
if __name__ == "__main__":
    print(os.getenv("GITHUB_TOKEN"))
    print(os.getenv("OPENAI_API_KEY"))
    utils.setup()
    # paste url:
    # "https://github.com/chiyeon/tmf-beat")
    author, repo_name = utils.get_repo_info(
        "https://github.com/Jingzhi-Su/PokerBot")
    print(author, repo_name)
    generateDataFile(author, repo_name, branch="main")
    allNames = generateFileNames()
    print(generateDescriptions(allNames))
    os.remove(storeFileName)
