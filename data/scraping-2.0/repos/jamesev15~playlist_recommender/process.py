from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from playlist.music import Music, Artist
from playlist.prompts import get_music_tags


def display_playlist(documents: list[Document]) -> str:
    music_infos = [
        Music.parse_obj(document.metadata).display_info()
        for document in documents
    ]
    return "\n".join(music_infos)


if __name__ == "__main__":
    vector_store = FAISS.load_local("vector_store", OpenAIEmbeddings())

    music_name = input("Enter the music name: ")
    artist = input("Enter the artist name: ")

    try:
        music_tags = get_music_tags(
            music=Music(name=music_name, artists=[Artist(name=artist)])
        )
    except:
        print("Try again or another music name")
    else:
        docs = vector_store.similarity_search(
            query=music_tags.display_info(), k=10
        )

        print(display_playlist(documents=docs))
