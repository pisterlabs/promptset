import io
import logging
import unittest
from os import environ

import PyPDF2
import requests
import typesense
from langchain.text_splitter import CharacterTextSplitter
from markdownify import markdownify as md
from typesense.exceptions import ObjectAlreadyExists
from bs4 import BeautifulSoup
import tiktoken
logger = logging.getLogger(__name__)
from more_itertools import chunked
import json

def extract_text_from_pdf_bytes(pdf_bytes):
    """
    Extract text from a PDF file represented as a byte object.

    :param pdf_bytes: Byte object of the PDF file
    :return: Extracted text as a string
    """
    text = ""
    try:
        # Create a file-like object from the bytes
        pdf_file = io.BytesIO(pdf_bytes)

        # Create a PDF reader object
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        # Iterate over each page and extract text
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"

        pdf_file.close()
    except Exception as e:
        # Handle exceptions
        text = f"An error occurred: {e}"

    return text

def clean_html(html_bytes):
    """
    Clean HTML content: remove non-HTML content and strip element attributes.

    :param html_bytes: Byte object of the HTML content
    :return: Cleaned HTML as a string
    """
    # Decode the bytes to a string
    html_string = html_bytes.decode('utf-8')

    # Parse the HTML content
    soup = BeautifulSoup(html_string, 'html.parser')

    # Remove script and style elements
    for script_or_style in soup(['script', 'style']):
        script_or_style.decompose()

    # Remove all attributes from HTML elements
    for tag in soup.find_all(True):
        tag.attrs = {}

    # Get the cleaned HTML as a string
    cleaned_html = str(soup)

    return cleaned_html

def convert_html_bytes_to_markdown(html_bytes):
    """
    Convert HTML content in byte format to Markdown.

    :param html_bytes: Byte object of the HTML content
    :return: Markdown formatted string
    """
    markdown_text = ""
    try:
        # Decode the bytes to a string
        html_string = clean_html(html_bytes)

        # Convert HTML to Markdown
        markdown_text = md(html_string,  convert=['b','h1','h2','h3','a'])
    except Exception as e:
        # Handle exceptions
        markdown_text = f"An error occurred: {e}"

    return markdown_text


def download_or_open(url_or_path):
    """
    Download a file from a URL or open a file from a path.

    :param url_or_path: URL or file path
    :return: File object
    """
    if url_or_path.startswith(('http', 'https')):
        # Download file from URL
        response = requests.get(url_or_path)
        response.raise_for_status()
        content_bytes = response.content
        content_type = response.headers.get('Content-Type', '').lower()
    else:
        with open(url_or_path, 'rb') as file:
            content_bytes = file.read()
        ext = url_or_path.split('.')[-1].lower()  # File extension

        if ext == 'pdf':
            content_type = 'application/pdf'
        elif ext in ['html', 'htm']:
            content_type = 'text/html'
        else:
            content_type = 'text/plain'

    # Convert content to text
    if content_type.startswith('application/pdf'):
        text = extract_text_from_pdf_bytes(content_bytes)
    elif content_type.startswith('text/html'):
        text = convert_html_bytes_to_markdown(content_bytes)
    else:
        # Assuming plain text for any other type
        text = content_bytes.decode('utf-8')

    return text


def move_toks(toks, pos, n):
    """Move forward or back number of lines for a given number of tokens"""

    direction = 1
    if n < 0:
        direction = -1
        n = -n

    while n > 0:
        pos += direction

        if pos >= len(toks):
            pos = len(toks)
            break
        elif pos < 0:
            pos = 0
            break

        n -= toks[pos]

    return pos

def process_content(text, chunk_size=500, chunk_overlap=50):
    """
    Process content from a URL or file path. Extracts and chunks text from PDF, HTML, or plain text.

    :param source: URL or file path
    :return: List of text chunks
    """
    # Determine if source is URL or file path

    # Chunking with tokens

    enc = tiktoken.encoding_for_model('gpt-4') # FIXME: use model_name

    # Some lines are really long, so break up the line if it is more than 100 tokens.
    lines = []
    max_tok_per_line = chunk_size // 10 # minlines per chunk
    for l in text.splitlines():
        if l.strip():
            toks = len(enc.encode(l))
            if toks < max_tok_per_line:
                lines.append(l)
            else:
                words = l.split()
                n_chunks = int(toks / max_tok_per_line)
                words_per_chunk = int(len(words) / n_chunks)
                chunks = chunked(words, words_per_chunk)
                for c in chunks:
                    lines.append(' '.join(c))

    toks = [len(enc.encode(e)) for e in lines]

    chunks = []
    # Chunk indexes
    start = 0
    end = 0

    while end < len(toks):
        end = move_toks(toks, start, chunk_size)
        chunks.append('\n'.join(lines[start:end]))
        start = move_toks(toks, end, -chunk_overlap)

    return chunks


class Library:
    def __init__(self, client: typesense.Client | None):
        self.client = client

    library_schema = {
        "name": "library",
        "fields": [
            {"name": "title", "type": "string"},
            {"name": "description", "type": "string", "optional": True},
            {"name": "source", "type": "string", "optional": True},
            {"name": "chunk", "type": "int32", "optional": True},
            {"name": "tags", "type": "string[]", "optional": True},
            {"name": "text", "type": "string"},
            {
                "name": "embedding",
                "type": "float[]",
                "embed": {
                    "from": ["text"],
                    "model_config": {
                        "model_name": "openai/text-embedding-ada-002",
                        "api_key": environ.get("OPENAI_API_KEY"),
                    },
                },
            },
        ],
    }

    def _create_collection(self, sch, delete=False):
        logger.debug(f"Loading {sch['name']} schema")

        try:
            self.client.collections.create(sch)
        except ObjectAlreadyExists:
            if delete:
                logger.debug(
                    f"Collection {sch['name']} already exists, deleting and recreating"
                )
                self.client.collections[sch["name"]].delete()
                self.client.collections.create(sch)
            else:
                logger.debug(f"Collection {sch['name']} already exists, skipping")
                return

    def create_collection(self, delete=False):
        """Create a collection with the given name and schema"""
        self._create_collection(self.library_schema, delete=delete)

    def clear_collection(self):
        """Clear the collection"""
        return self.client.collections["library"].documents.delete({"filter_by": "chunk:>=0"})

    def list(self):
        """List all documents in the collection"""
        r = self.client.collections["library"].documents.export()
        for e in r.splitlines():
            d = json.loads(e)
            del d['embedding']

            d = {key: d[key] for key in sorted(d.keys())}

            yield d

    def count(self):
        return len(list(self.list()))

    def _add_document(self, doc):

        doc = {k: v for k, v in doc.items() if v is not None}

        if "id" in doc:
            r = self.client.collections["library"].documents.upsert(id)
        else:
            r = self.client.collections["library"].documents.create(doc)

        if 'embedding' in r:
            del r['embedding']

        if len(r['text']) > 200:
            r['text'] = '...' + r['text'][-200:]

        return r

    # Add a document from function arguments
    def add_document(self, title: str = None, text: str = None,
                     description=None, source=None, chunk=None, tags=None):
        """
        Add a document to the search system by providing either a text or a source URL.
        If the text is empty, the source will be used as a path or URL to load the document

        :param title: Title of the item (string)
        :param description: Description of the item (string)
        :param source: Source of the item (string, optional). If this is a URL,
            or a path the content will be downloaded from the source.
        :param chunk: Chunk number (int, optional)
        :param tags: List of tags (list of strings, optional)
        :param text: Text content (string). Option. If not provided, the
            content will be downloaded from the source.
        :return: Dictionary with non-None arguments
        """

        if text is None and source is not None:
            text = download_or_open(source)

        assert text is not None or source is not None, "Must provide either text or source"

        chunks = process_content(text)

        title = title or chunks[0].splitlines()[0]

        from tqdm import tqdm

        docs = []
        for chunk_n, chunk in enumerate(chunks):
            doc = {
                "title": title,
                "description": description,
                "source": source,
                "chunk": chunk_n,
                "tags": tags,
                "text": chunk
            }
            docs.append(doc)

        r = self.client.collections["library"].documents.import_(docs)

        return {"title": title, 'source': source, "chunks": len(docs)}


    # noinspection PyUnusedLocal
    def _search(self, text, tags=None):
        """Search the collection for a given query"""

        query = {"q": text,
                 "query_by": "description, embedding",
                 "prefix": False,
                 "exclude_fields": "embedding"}

        r = self.client.collections["library"].documents.search(query)

        return r

    # noinspection PyUnusedLocal
    def search(self, text, tags=None):

        r = self._search(text)

        hits = []
        # Consoldate some of the results fields
        for h in r['hits']:
            tmi = h['text_match_info']
            tmi['rank_fusion_score'] = h['hybrid_search_info']['rank_fusion_score']
            del h['hybrid_search_info']
            tmi['text_match'] = h['text_match']
            del h['text_match']
            tmi['vector_distance'] = h['vector_distance']
            del h['vector_distance']

            doc = h['document']
            doc['_text_match_info'] = tmi

            hits.append(doc)

        return hits

    def get_document(self, doc_id):
        d = self.client.collections['library'].documents[doc_id].retrieve()
        if 'embedding' in d:
            del d['embedding']

        return d


class TestCase(unittest.TestCase):

    def setUp(self):
        self.client = typesense.Client(
            {
                "api_key": "xyz",
                "nodes": [{"host": "barker", "port": "8108", "protocol": "http"}],
                "connection_timeout_seconds": 20,
            }
        )

    def test_basic(self):
        ts = Library(self.client)

        ts.create_collection(delete=True)

        ts._add_document(
            {
                "title": "Software Engineer",
                "description": "I am a software engineer.",
                "tags": ["software", "engineer"],
                "text": "I am a software engineer.",
            }
        )

        ts._add_document(
            {
                "title": "PockiWoki",
                "description": "Just nonsense",
                "tags": ["software", "engineer"],
                "text": "Small rusted rabbits knit furiously in the night.",
            }
        )

        # noinspection PyUnusedLocal
        def p(resp):
            for e in resp["hits"]:
                d = e["document"]
                del d["embedding"]
                print(e["vector_distance"], d)

        r = ts.search("do more web programming")
        print(r)
        self.assertEqual(r[0]['title'], "Software Engineer")

        r = ts.search("rodents make sweaters")
        self.assertEqual(r[0]['title'], "PockiWoki")

    def test_add_doc(self):
        ts = Library(self.client)
        ts.create_collection(delete=True)

        ts.add_document('Drinking Whiskey', 'Wishkey will make you more inteligent')
        ts.add_document('Cleaning your Ears',
                        'If you don\'t clean your ears, you will get cancer',
                        description='A medical article of grave importance to audiophiles')

        r = ts.search("How to prevent fatal diseases")
        import json
        print(json.dumps(r, indent=4))

    def test_doc_loader(self):

        text = download_or_open('http://eric.busboom.org/wp-content/uploads/sites/9/2023/11/Unicycle-Lesson-Business.pdf')

        self.assertEqual(len(text), 732)

        text = download_or_open('/Volumes/Cache/Downloads/Unicycle Lesson Business.pdf')
        self.assertEqual(len(text), 732)
        text = text.replace("\n", " ")
        self.assertTrue('Study unicycle lessons' in text)

        text = download_or_open('https://ericbusboom.com')
        self.assertEqual(len(text), 4349)

    def test_tok_moves(self):

        toks = [1]*100
        pos = 50

        self.assertEqual(60, move_toks(toks, pos, 10))

        self.assertEqual(40, move_toks(toks, pos, -10))

    def test_doc_splitter(self):
        from textwrap import fill

        byts, typ = download_or_open('https://en.wikipedia.org/wiki/Tartan')

        l = process_content(byts, typ)
        self.assertEqual(len(l),  117)

        #for w in l:
        #    print("===============")
        #    print(fill(w, 100))


    def test_chunked_load(self):
        import json

        ts = Library(self.client)
        #ts.create_collection(delete=True)
        #r = ts.add_document(source='https://en.wikipedia.org/wiki/Tartan')

        r = ts.search("How are older tartans different from modern one")
        print(json.dumps(r, indent=4))

if __name__ == "__main__":
    unittest.main()
