from langchain.text_splitter import MarkdownTextSplitter

markdown_text = """
# 

# Welcome to My Blog!

## Introduction
Hello everyone! My name is **John Doe** and I am a _software developer_. I specialize in Python, Java, and JavaScript.

Here's a list of my favorite programming languages:

1. Python
2. JavaScript
3. Java

You can check out some of my projects on [GitHub](https://github.com).

## About this Blog
In this blog, I will share my journey as a software developer. I'll post tutorials, my thoughts on the latest technology trends, and occasional book reviews.

Here's a small piece of Python code to say hello:

\``` python
def say_hello(name):
    print(f"Hello, {name}!")

say_hello("John")
\```

Stay tuned for more updates!

## Contact Me
Feel free to reach out to me on [Twitter](https://twitter.com) or send me an email at johndoe@email.com.

"""

markdown_splitter = MarkdownTextSplitter(chunk_size=100, chunk_overlap=0)
docs = markdown_splitter.create_documents([markdown_text])

print(docs)
