import re
import pysrt
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyMuPDFLoader, Docx2txtLoader, YoutubeLoader, WebBaseLoader, TextLoader
from langchain.schema import Document
from tempfile import NamedTemporaryFile
import logging
logger = logging.getLogger(__name__)

class InfoLoader():
    def __init__(self, config):
        '''
        Class for handling all data extraction and chunking
        Inputs:
            config - dictionary from yaml file, containing all important parameters
        '''
        self.config = config
        self.remove_leftover_delimiters = config['splitter_options']['remove_leftover_delimiters']

        # Main list of all documents
        self.document_chunks_full = []
        self.document_names = []

        if config['splitter_options']['use_splitter']:
            if config['splitter_options']['split_by_token']:
                self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                    chunk_size=config['splitter_options']['chunk_size'],
                    chunk_overlap=config['splitter_options']['chunk_overlap'],
                    separators = config['splitter_options']['chunk_separators']
                    )
            else:
                self.splitter = RecursiveCharacterTextSplitter(
                    chunk_size=config['splitter_options']['chunk_size'],
                    chunk_overlap=config['splitter_options']['chunk_overlap'],
                    separators = config['splitter_options']['chunk_separators']
                    )
        else:
            self.splitter = None
        logger.info('InfoLoader instance created')

    def get_chunks(self, uploaded_files, weblinks):
        # Main list of all documents
        self.document_chunks_full = []
        self.document_names = []
        
        def remove_delimiters(document_chunks : list):
            '''
            Helper function to remove remaining delimiters in document chunks
            '''
            for chunk in document_chunks:
                for delimiter in self.config['splitter_options']['delimiters_to_remove']:
                    chunk.page_content = re.sub(delimiter, ' ', chunk.page_content)
            return document_chunks

        def remove_chunks(document_chunks : list):
            '''
            Helper function to remove any unwanted document chunks after splitting
            '''
            front = self.config['splitter_options']['front_chunk_to_remove']
            end = self.config['splitter_options']['last_chunks_to_remove']
            # Remove pages
            for _ in range(front):
                del document_chunks[0]
            for _ in range(end):
                document_chunks.pop()
                logger.info(f'\tNumber of pages after skipping: {len(document_chunks)}')
            return document_chunks

        def get_pdf(temp_file_path : str, title : str):
            '''
            Function to process PDF files
            '''
            loader = PyMuPDFLoader(temp_file_path) #This loader preserves more metadata

            if self.splitter:
                document_chunks = self.splitter.split_documents(loader.load())
            else:
                document_chunks = loader.load()

            if 'title' in document_chunks[0].metadata.keys():
                title = document_chunks[0].metadata['title']

            logger.info(f"\t\tOriginal no. of pages: {document_chunks[0].metadata['total_pages']}")

            return title, document_chunks

        def get_txt(temp_file_path : str, title : str):
            '''
            Function to process TXT files
            '''
            loader = TextLoader(temp_file_path, autodetect_encoding=True)

            if self.splitter:
                document_chunks = self.splitter.split_documents(loader.load())
            else:
                document_chunks = loader.load()

            # Update the metadata
            for chunk in document_chunks:
                chunk.metadata['source'] = title
                chunk.metadata['page'] = 'N/A'

            return title, document_chunks

        def get_srt(temp_file_path : str, title : str):
            '''
            Function to process SRT files
            '''
            subs = pysrt.open(temp_file_path)

            text = ''
            for sub in subs:
                text += sub.text
            document_chunks = [Document(page_content=text)]
            
            if self.splitter:
                document_chunks = self.splitter.split_documents(document_chunks)

            # Update the metadata
            for chunk in document_chunks:
                chunk.metadata['source'] = title
                chunk.metadata['page'] = 'N/A'

            return title, document_chunks

        def get_docx(temp_file_path : str, title : str):
            '''
            Function to process DOCX files
            '''
            loader = Docx2txtLoader(temp_file_path)

            if self.splitter:
                document_chunks = self.splitter.split_documents(loader.load())
            else:
                document_chunks = loader.load()

            # Update the metadata
            for chunk in document_chunks:
                chunk.metadata['source'] = title
                chunk.metadata['page'] = 'N/A'

            return title, document_chunks

        def get_youtube_transcript(url : str):
            '''
            Function to retrieve youtube transcript and process text
            '''
            loader = YoutubeLoader.from_youtube_url(
                url, 
                add_video_info=True,
                language=["en"],
                translation="en"
            )

            if self.splitter:
                document_chunks = self.splitter.split_documents(loader.load())
            else:
                document_chunks = loader.load_and_split()   

            # Replace the source with title (for display in st UI later)
            for chunk in document_chunks:
                chunk.metadata['source'] = chunk.metadata['title']
            title = chunk.metadata['title']
            logger.info(title)

            return title, document_chunks

        def get_html(url : str):
            '''
            Function to process websites via HTML files
            '''
            loader = WebBaseLoader(url)

            if self.splitter:
                document_chunks = self.splitter.split_documents(loader.load())
            else:
                document_chunks = loader.load_and_split()
            
            title = document_chunks[0].metadata['title']
            logger.info(title)

            return title, document_chunks

        # Handle file by file
        for file_index, file in enumerate(uploaded_files):

            # Get the file type and file name
            file_type = file.name.split('.')[-1].lower()
            logger.info(f'\tSplitting file {file_index+1} : {file.name}')
            file_name = ''.join(file.name.split('.')[:-1])

            with NamedTemporaryFile(delete=False, suffix=f".{file_type}") as temp_file:
                temp_file_path = temp_file.name
                temp_file.write(file.read())

            # Handle different file types
            if file_type =='pdf':
                title, document_chunks = get_pdf(temp_file_path, file_name)
            elif file_type == 'txt':
                title, document_chunks = get_txt(temp_file_path, file_name)
            elif file_type == 'docx':
                title, document_chunks = get_docx(temp_file_path, file_name)
            elif file_type == 'srt':
                title, document_chunks = get_srt(temp_file_path, file_name)

            # Additional wrangling - Remove leftover delimiters and any specified chunks
            if self.remove_leftover_delimiters:
                document_chunks = remove_delimiters(document_chunks)
            if self.config['splitter_options']['remove_chunks']:
                document_chunks = remove_chunks(document_chunks)

            logger.info(f'\t\tExtracted no. of chunks: {len(document_chunks)}')
            self.document_names.append(title)
            self.document_chunks_full.extend(document_chunks)

        # Handle youtube links:
        if weblinks[0] != '':
            logger.info(f'Splitting weblinks: total of {len(weblinks)}')
            
            # Handle link by link
            for link_index, link in enumerate(weblinks):
                logger.info(f'\tSplitting link {link_index+1} : {link}')
                if 'youtube' in link:
                    title, document_chunks = get_youtube_transcript(link)
                else:
                    title, document_chunks = get_html(link)

                # Additional wrangling - Remove leftover delimiters and any specified chunks
                if self.remove_leftover_delimiters:
                    document_chunks = remove_delimiters(document_chunks)
                if self.config['splitter_options']['remove_chunks']:
                    document_chunks = remove_chunks(document_chunks)

                print(f'\t\tExtracted no. of chunks: {len(document_chunks)}')
                self.document_names.append(title)
                self.document_chunks_full.extend(document_chunks)
          
        logger.info(f'\tNumber of document chunks extracted in total: {len(self.document_chunks_full)}\n\n')

    

    

        
        
