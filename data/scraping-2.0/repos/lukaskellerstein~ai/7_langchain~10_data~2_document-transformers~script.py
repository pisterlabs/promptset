from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    Language,
)

# load a file
with open("./source/forrest-gump.txt") as f:
    book = f.read()


# ---------------------
# TEXT Splitter
#
# split by ["\n\n", "\n", " ", ""]
# ---------------------
text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    add_start_index=True,
)

result = text_splitter.create_documents([book])
print(result)


# ---------------------
# Split by character
#
# split by custom separator
# ---------------------
text_splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)

result = text_splitter.create_documents([book])
print(result)

# ---------------------
# Split by code [JS, Python, ...etc.]
# ---------------------

# Python
PYTHON_CODE = """
def hello_world():
    print("Hello, World!")

# Call the function
hello_world()
"""
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=50, chunk_overlap=0
)
result = python_splitter.create_documents([PYTHON_CODE])
print(result)

# JS
JS_CODE = """
function helloWorld() {
  console.log("Hello, World!");
}

// Call the function
helloWorld();
"""

js_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.JS, chunk_size=60, chunk_overlap=0
)
result = js_splitter.create_documents([JS_CODE])
print(result)

# ---------------------
# HTML
# ---------------------
html_text = """
<!DOCTYPE html>
<html>
    <head>
        <title>🦜️🔗 LangChain</title>
        <style>
            body {
                font-family: Arial, sans-serif;
            }
            h1 {
                color: darkblue;
            }
        </style>
    </head>
    <body>
        <div>
            <h1>🦜️🔗 LangChain</h1>
            <p>⚡ Building applications with LLMs through composability ⚡</p>
        </div>
        <div>
            As an open source project in a rapidly developing field, we are extremely open to contributions.
        </div>
    </body>
</html>
"""

html_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.MARKDOWN, chunk_size=60, chunk_overlap=0
)
result = html_splitter.create_documents([html_text])
print(result)
