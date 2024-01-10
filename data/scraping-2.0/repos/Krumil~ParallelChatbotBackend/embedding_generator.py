import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import GitbookLoader

DEPLOYMENT_ENV = os.environ.get('DEPLOYMENT_ENV', 'DEVELOPMENT')
if DEPLOYMENT_ENV == 'PRODUCTION':
	# these lines swap the stdlib sqlite3 lib with the pysqlite3 package
	__import__('pysqlite3')
	import sys
	sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')	

def create_embeddings():
	if DEPLOYMENT_ENV == 'PRODUCTION':
		base_directory = "/var/data/embeddings/"
	else:
		base_directory = ".\\embeddings\\"

	splits = []    
	pdfs = [
		"documents/main.md",
		"documents/renown.md"
	]

	csvs = [
		"documents/cards.csv"
	]

	gitbooks = [
		'https://docs.echelon.io/'
	]

	for document in pdfs:
		with open(document, "r") as f:
			markdown_document = f.read()

		headers_to_split_on = [
			("##", "section"),
			("###", "topic"),
		]

		markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
		md_header_splits = markdown_splitter.split_text(markdown_document)    
		splits.extend(md_header_splits)

	pdf_vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(), persist_directory=os.path.join(base_directory, "pdf_chroma_db"))
	pdf_vectorstore.persist()

	data_csv = []
	for document in csvs:
		csv_loader = CSVLoader(file_path=document, encoding="utf-8")
		data = csv_loader.load_and_split()
		data_csv.extend(data)

	csv_vectorstore = Chroma.from_documents(documents=data_csv, embedding=OpenAIEmbeddings(), persist_directory=os.path.join(base_directory, "csv_chroma_db"))
	csv_vectorstore.persist()

	data_gitbook = []
	for document in gitbooks:
		gitbook_loader = GitbookLoader(document, load_all_paths=True)
		data = gitbook_loader.load_and_split()
		data_gitbook.extend(data)

	gitbook_vectorstore = Chroma.from_documents(documents=data_gitbook, embedding=OpenAIEmbeddings(), persist_directory=os.path.join(base_directory, "gitbook_chroma_db"))
	gitbook_vectorstore.persist()
