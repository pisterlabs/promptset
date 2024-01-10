from docx import Document
import openai
import tiktoken
import os
import shutil
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from openai.embeddings_utils import get_embedding, cosine_similarity

load_dotenv('.env')
openai.api_key=os.environ.get('OPENAI_API_KEY')

# Embedding model. Currently recommended version. 
EMBEDDING_MODEL = 'text-embedding-ada-002'
# Model used for the chat generation.
GPT_MODEL = 'gpt-3.5-turbo'
CHAT_COMPLETION_CTX_LENGTH = 4097
# Define the query length in order to subtract from the maximum length the chunk can contain. 
QUERY_LENGTH = 200
EMBEDDING_ENCODING = 'cl100k_base'

class Preprocessor:
    """
    This class is used for preprocessing DOCX documents and chunking them into separate text files.
    """
    def __init__(self, docx_file):
        """
        Initialize the Preprocessor object.

        Args:
            docx_file (str): Path to the DOCX file to be processed.

        """
        self.docx_file = docx_file

    def convert_docx_to_txt(self, txt_file='data/sustainability_report.txt'):
        """
        Convert the DOCX file to TXT format.

        Args:
            txt_file (str): Path to the output TXT file.

        Returns:
            None

        """
        # Open the DOCX file
        doc = Document(self.docx_file)

        # Extract the text from paragraphs
        text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])

        # Write the text to a TXT file
        with open(txt_file, 'w', encoding='utf-8') as file:
            file.write(text)
    
    def preprocess(self):
        """
        Preprocess the data.

        The preprocessing pipeline can be expanded by adding additional function calls.

        Returns:
            None

        """
        print(f'Preprocessing of {self.docx_file} initiated...')
        self.convert_docx_to_txt()
        print('Preprocessing done.')

class TextEmbedder:
    """
    This class is used for embedding TXT files. 
    If the TXT file is larger than the maximum token length specified, it will be chunked in smaller txt files.  
    """
    def __init__(self, txt_file, output_dir):
        self.txt_file = txt_file

        self.output_dir = output_dir

    def chunk_text(self, encoding_name, max_token_length):
        """
        Chunk the text into smaller segments based on the maximum token length.

        Args:
            encoding_name (str): Name of the encoding to use for tokenization.
            max_token_length (int): Maximum length of tokens allowed in each chunk.

        Returns:
            None

        """
        # Open the text file for reading
        with open(self.txt_file, 'r', encoding='utf-8') as file:
            text = file.read()

            # Get the encoding and the strings in token lengths
            encoding = tiktoken.get_encoding(encoding_name)

            # Convert the text into tokens using the specified encoding
            encoded_text = encoding.encode(text)

            # Split the text into chunks based on the maximum token length
            encoded_chunks = [encoded_text[i:i + max_token_length] for i in range(0, len(encoded_text), max_token_length)]

            # Overwrite the directory if it exists
            if os.path.exists(self.output_dir):
                shutil.rmtree(self.output_dir)

            # Create the output directory
            os.makedirs(self.output_dir, exist_ok=True)

            # Iterate over the encoded chunks and write them to separate files
            for i, chunk in enumerate(encoded_chunks):
                print(f'Chunk {i} is of length: {len(chunk)}\n')
                # Decode the chunk into text
                decoded_text = encoding.decode(chunk)
                output_file = os.path.join(self.output_dir, f'report_chunk_{i}.txt')
                with open(output_file, 'w', encoding='utf-8') as file:
                    file.write(decoded_text)  

    def embed_chunks(self, csv_embeddings_file=None, vector_db=None):
        """
        Embeds the chunks of text and saves the embeddings with the corresponding text to a CSV file.

        Args:
            csv_embeddings_file (str): Path to the output CSV file to save the embeddings.

        Returns:
            None

        """
        # Create an empty DataFrame to store the text and embeddings
        df = pd.DataFrame(columns=["text", "embedding"])

        # Iterate over the files in the output directory
        for filename in os.listdir(self.output_dir):
            file_path = os.path.join(self.output_dir, filename)


            # Check if the file is a regular file
            if os.path.isfile(file_path):
                # Open the file and read the text
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()

                    # Generate the embedding for the text using the OpenAI Embedding API
                    embedding = openai.Embedding.create(
                        input=text,
                        model=EMBEDDING_MODEL
                    )['data'][0]['embedding']

                    # Append the text and embedding to the DataFrame if using a csv for storing the embeddings.
                    if csv_embeddings_file:
                        df = df.append({'text': text, 'embedding': embedding}, ignore_index=True)

                    # Save the text and the embedding in the vector database
                    if vector_db:
                        vector_db.upsert([(filename, embedding)])

        # Save the DataFrame to a CSV file
        if csv_embeddings_file:
            df.to_csv(csv_embeddings_file, index=False, mode='w')

class QueryEngine():
    def search_chunks(self, csv_embeddings_file, query, k=3, pprint=True):
        """
        This function provides semantic search using embeddings.
        Search the chunks and find the k most similar chunks based on the query.

        Args:
            csv_embeddings_file (str): Path to the embeddings file.
            query (str): The query string used for similarity search.
            k (int, optional): The number of most similar chunks to retrieve. Defaults to 3.
            pprint (bool, optional): Whether to print the results. Defaults to True.

        Returns:
            pandas.DataFrame: The DataFrame containing the k most similar chunks.
        """
        # Read the embeddings file into a DataFrame
        df = pd.read_csv(csv_embeddings_file)

        # Convert the 'embedding' column from string to numpy array
        df["embedding"] = df.embedding.apply(eval).apply(np.array)

        # Get the embedding for the query
        query_embedding = get_embedding(
            query,
            engine=EMBEDDING_MODEL
        )

        # Apply cosine similarity to find the k most similar chunks
        df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, query_embedding))

        # Sort the DataFrame by similarity in descending order and retrieve the top k chunks
        results = (
            df.sort_values("similarity", ascending=False)
            .head(k)
        )
        
        # Print the results if pprint is True
        if pprint:
            for r in results:
                print(r[:200])
                print()

        return results
    
    def query(self, df, user_message, max_num_sentences, max_num_words):
        """
        Queries the knowledge base by providing the user message and concatenating the relevant text
        from the knowledge base. 

        Args:
            df (pandas dataframe): contains the embeddings and texts as a dataframe.
            
        """
        # concatenate the texts from the df together. 
        concatenated_kb = df['text'].str.cat(sep="Chunk: ")

        print('The concatenated knowledge base: \n', concatenated_kb)

        # System message
        role_content = f"""
        Answer the user query using information from the knowledge based provided as chunks.
        Limit your answer to {max_num_sentences} sentences and up to {max_num_words} words in length.
        """

        # Delimiter between the user message and the knowledge base. 
        delimiter = "Below are the chunks used as the knowledge based:\n"

        messages = [{"role": "system", "content": role_content},
            {"role": "user", "content": user_message + delimiter + concatenated_kb}]
        
        print('The user content: ', user_message + delimiter + concatenated_kb)

        response_message = openaiChatCompletion(messages)
        return response_message

def openaiChatCompletion(messages):
    response = openai.ChatCompletion.create(
            model = GPT_MODEL, 
            messages=messages
        )
    response_message = response["choices"][0]["message"]
    return response_message