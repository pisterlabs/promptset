from langchain.llms import Ollama

ollama = Ollama(model="llama2:7b", temperature=0)


def classify_book(book_title):
    system_prompt = """
    'Pride and Prejudice' = Romance
    'The Lord of the Rings' = Fantasy
    'The Great Gatsby' = Fiction
    '1984' = Dystopian
    ###
    """
    return ollama.invoke(f"{system_prompt}\n'{book_title}' = ")


input_title = input("Give me a book title: ")
print(classify_book(input_title))
