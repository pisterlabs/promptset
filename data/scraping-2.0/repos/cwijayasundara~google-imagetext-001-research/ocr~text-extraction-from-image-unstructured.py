from langchain.document_loaders.image import UnstructuredImageLoader

loader = UnstructuredImageLoader("/Users/chamindawijayasundara/Documents/self_learn/palm2-multimodalembedding/text"
                                 "-extraction-from-image/images/img.png")

data = loader.load()

print(data[0])

