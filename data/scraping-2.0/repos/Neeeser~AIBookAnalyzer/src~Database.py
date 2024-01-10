import sqlite3
import os
from langchain import FAISS
from langchain.document_loaders import TextLoader, PyPDFLoader, UnstructuredEPubLoader
from langchain.embeddings import GPT4AllEmbeddings
from langchain.text_splitter import CharacterTextSplitter
import hashlib
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup


class VectorDatabase:
    def __init__(self, embeddings=GPT4AllEmbeddings()):
        self.embeddings = embeddings

    def create_index(self, index_name, split_text):
        db = None
        if isinstance(split_text[0], str):
            db = FAISS.from_texts(split_text, self.embeddings)
        else:
            db = FAISS.from_documents(split_text, self.embeddings)
        db.save_local("faiss_db", index_name=index_name)

    def load_index(self, index_name):
        db = FAISS.load_local(
            "faiss_db", index_name=index_name, embeddings=self.embeddings
        )
        return db

    def split_text(self, loader, chunk_size=1000, chunk_overlap=0):
        text_splitter = CharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        if isinstance(loader, str):
            return text_splitter.split_text(loader)

        documents = text_splitter.split_documents(loader)
        return documents

    def add_txt(self, document_path):
        raw_documents = TextLoader(document_path)
        return self.split_text(raw_documents.load())

    def add_pdf(self, pdf_path):
        loader = PyPDFLoader(pdf_path)
        return self.split_text(loader.load())

    def add_epub(self, epub_path):
        book = epub.read_epub(epub_path)
        text = ""
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                soup = BeautifulSoup(item.get_content(), "html.parser")
                text += soup.get_text()
        loader = text
        return self.split_text(loader)


class BookDatabase:
    def __init__(self, db_file):
        self.conn = sqlite3.connect(db_file)
        self.conn.row_factory = sqlite3.Row
        self.cursor = self.conn.cursor()
        self.create_table()

    def create_table(self):
        self.cursor.execute(
            """CREATE TABLE IF NOT EXISTS books
                               (id INTEGER PRIMARY KEY AUTOINCREMENT,
                                title TEXT,
                                author TEXT,
                                index_name TEXT,
                                filename TEXT)"""
        )
        self.conn.commit()

    def search_books(self, keyword):
        self.cursor.execute(
            "SELECT * FROM books WHERE title LIKE ? OR author LIKE ?",
            (f"%{keyword}%", f"%{keyword}%"),
        )
        rows = self.cursor.fetchall()

        # Convert the rows to dictionaries
        books = [dict(row) for row in rows]
        return books

    def add_book(self, title, author, index_name, filename):
        # Check if the book already exists in the database
        self.cursor.execute(
            "SELECT * FROM books WHERE title = ? AND author = ? AND filename = ?",
            (title, author, filename),
        )
        existing_book = self.cursor.fetchone()
        if existing_book:
            print("Book already exists in the database.")
            return existing_book[0]  # Return the ID of the existing book

        # Insert the new book entry if it doesn't exist
        self.cursor.execute(
            "INSERT INTO books (title, author, index_name, filename) VALUES (?, ?, ?, ?)",
            (title, author, index_name, filename),
        )
        self.conn.commit()
        return self.cursor.lastrowid

    # Checks if book exists in database by filename
    def check_book(self, filename):
        self.cursor.execute("SELECT * FROM books WHERE filename = ?", (filename,))
        existing_book = self.cursor.fetchone()
        if existing_book:
            return True
        else:
            return False

    def delete_all_books(self):
        self.cursor.execute("DELETE FROM books")
        self.conn.commit()
        print("All books deleted.")

    def delete_book_by_id(self, book_id):
        self.cursor.execute("DELETE FROM books WHERE id = ?", (book_id,))
        self.conn.commit()
        print(f"Book with ID {book_id} deleted.")

    def close_connection(self):
        self.cursor.close()
        self.conn.close()

    def get_book_by_id(self, book_id):
        self.cursor.execute("SELECT * FROM books WHERE id = ?", (book_id,))
        row = self.cursor.fetchone()
        if row:
            return dict(row)
        else:
            return None

    def print_all_books(self):
        self.cursor.execute("SELECT * FROM books")
        rows = self.cursor.fetchall()
        for row in rows:
            print(str(row))


class Database:
    def __init__(self):
        self.db_file = "books.db"
        self.vdb = VectorDatabase()
        self.bdb = BookDatabase(self.db_file)
        self.supported_extensions = [".txt", ".pdf", ".epub"]

    def load_all_bookdata(self):
        # For all books located in bookdata/
        for filename in os.listdir("bookdata"):
            # Get the file extension
            extension = os.path.splitext(filename)[1]
            # Check if the file extension is supported
            if extension in self.supported_extensions:
                # Get the file path
                filepath = os.path.join("bookdata", filename)
                # Check if the book is already in the BookDatabase
                if self.bdb.check_book(filename):
                    print(f"{filename} already exists in the database.")
                    continue
                document = None
                # Vectorize the book
                if extension == ".txt":
                    documents = self.vdb.add_txt(filepath)
                elif extension == ".pdf":
                    documents = self.vdb.add_pdf(filepath)
                elif extension == ".epub":
                    documents = self.vdb.add_epub(filepath)
                else:
                    print(f"{filename} is not a supported file type.")
                    continue

                # Create the index
                index_hash = self.create_index_hash(filename)
                self.vdb.create_index(index_hash, documents)

                # Extract author from inside {} in filename
                author = filename.split("{")[1].split("}")[0]
                # Extract title from filename
                title = filename.split("{")[0].strip()
                # Add the book to the BookDatabase
                self.bdb.add_book(title, author, index_hash, filename)

    def create_index_hash(self, book_file_name):
        # Hash the file name to create a unique index name
        index_name = hashlib.sha256(book_file_name.encode()).hexdigest()
        return index_name
