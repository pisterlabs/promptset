from VectorDatabase import VectorDatabase
import OpenAIClient

##########################################################################
# Run the RAG system
##########################################################################

def main():
    print("Starting system...")
    # Load the database to use
    database = VectorDatabase(path="./chromadb_demo")
    client = database.getClient()
    embedding_function = database.getSentenceTransformerEmbeddingFunction('sentence-transformers/all-mpnet-base-v2')
    collection = client.get_or_create_collection("chars1000-50_mpnet-base", embedding_function=embedding_function)
    print("Welcome to the GoodReads book suggestion system!")
    while(True):
        # Get the user query
        query = input("Please enter your book query:").strip()
        if query == "quit" or query == "exit" or query == "q":
            exit()
        print("\nWhile searching for a book, may I offer you some tea or coffee?")
        # Ask OpenAI to define a genre
        genre = OpenAIClient.getGenre(query)
        if genre != "Unspecified" and query.find(genre) == -1:
            print("A fitting genre would be", genre)
        # Ask OpenAI to define an Author
        author = OpenAIClient.getAuthor(query)
        if author != "Unspecified":
            print("Your asking author is", author)
        # Ask OpenAI to split or rephrase the query
        if len(query) > 200:
            query, query_2 = OpenAIClient.split(query)
            print("\nWe will split your query to:")
            print("-", query)
            print("-", query_2)
        else:
            # Ask OpenAI to rephrase the query
            query_2 = OpenAIClient.rephrase(query)
            print("\nWe also use the following query:")
            print("-", query_2)
        # Get 10 Suggested Chunks
        books = database.query(
            collection=collection,
            query_texts=[query],
            n_results=10
        )
        # Get 10 Suggested Chunks
        books_2 = database.query(
            collection=collection,
            query_texts=[query_2],
            n_results=10
        )
        # Select the books with the higher rankings
        books += books_2
        books_2 = []
        for book in books:
            books_2.append([book['distance'], book])
        books_2 = sorted(books_2, key=lambda x: x[0], reverse=True)
        books = []
        for book in books_2:
            if book[1] in books:
                continue
            else:
                books.append(book[1])
            if len(books) >= 10:
                break
        # Ask OpenAI to select a Chunk
        response = OpenAIClient.selectBook(books, query)
        openAI_content = response.choices[0].message.content
        # Find the provided chunk
        book = None
        for book in books:
            if openAI_content.find(book['title']) > -1 and query.find(book['title']) == -1:
                break
        print("")
        print(response.choices[0].message.content)
        # Provide the Book & Summary (or URL) to the User
        print("\n Some input from our reviewers for this book is:")
        print("\n", book['content'])
        print("\n You may see more details here:", book['url'])
        print("")

if __name__ == '__main__':
    main()
