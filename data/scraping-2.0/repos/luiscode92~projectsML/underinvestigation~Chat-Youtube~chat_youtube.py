from langchain.document_loaders import YoutubeLoader
from langchain.indexes import VectorstoreIndexCreator
loader = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=q6Tsz9Ss6_g",  language='es')
index = VectorstoreIndexCreator().from_loaders([loader])
query = "de que trata el video?"
index.query(query)