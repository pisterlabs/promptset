from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
import langchain


langchain.verbose = True

# loader = DirectoryLoader("../langchain/docs/_build/html/", glob="**/*.html")
loader = DirectoryLoader("../demo/", glob="*.html")
index = VectorstoreIndexCreator().from_loaders([loader])
print("index created")

result = index.query("呪術廻戦の概要を1文で説明してください。")
print(f"result: {result}")
