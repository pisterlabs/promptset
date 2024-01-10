from langchain.document_loaders import SeleniumURLLoader
urls = [
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://goo.gl/maps/NDSHwePEyaHMFGwh8",
]

loader = SeleniumURLLoader(urls=urls)

data = loader.load()
print(data)