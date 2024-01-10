import dotenv
dotenv.load_dotenv()

from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import BeautifulSoupTransformer

# Load HTML
loader = AsyncChromiumLoader(["https://www.comohotels.com/destinations"])
html = loader.load()

# Transform
bs_transformer = BeautifulSoupTransformer()
docs_transformed = bs_transformer.transform_documents(html, tags_to_extract=["span"])


from langchain.document_loaders import AsyncHtmlLoader

urls = ["https://www.comohotels.com/destinations"]
loader = AsyncHtmlLoader(urls)
docs = loader.load()
