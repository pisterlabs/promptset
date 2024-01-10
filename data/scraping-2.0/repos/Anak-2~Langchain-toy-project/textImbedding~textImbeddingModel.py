import dotenv
from langchain.embeddings import OpenAIEmbeddings

dotenv_file = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv_file)

OpenAIEmbeddings()
