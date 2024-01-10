'''
Author: Utpal Kumar, BSL, UCB
Date: 2023-04-27
Email: utpalkumar@berkeley.edu
'''
from PyPDF2 import PdfReader
import pickle
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
import os, sys
import yaml
import uuid

import argparse


# Create the parser
parser = argparse.ArgumentParser(description='Read a paper and answer questions about it.')

# Add arguments
parser.add_argument('-c','--configfile', type=str, help='the path to the paper')
parser.add_argument('-q','--query', type=str, help='the query to ask about the paper')


# Parse the arguments
args = parser.parse_args()

with open(args.configfile, 'r') as f:
    config = yaml.safe_load(f)
# print(args)

openaikey = os.environ["OPENAI_API_KEY"]


########################    MAIN    ##############################
def main():
    paperFile = config['paper_path']

    if not os.path.exists(paperFile):
        sys.exit("Error: Paper file does not exist")

    paperreader = PaperReader(config)
    # docsearch, chain = create_model(paperFile)

    ## Query the document
    query = args.query
    out = paperreader.query_document(query)
    if config['document_output']:
        outdoc = paperFile.replace(".pdf", "_output.md")
        if config['clear_cache']:
            if os.path.exists(outdoc):
                os.remove(outdoc)
        with open(outdoc, 'a') as f:
            # f.write("====================\n")
            f.write("="*100+"\n")
            f.write("QUERY: {}\n".format(query))
            f.write("OUTPUT: {}\n".format(out))
            f.write("\n")
            # f.write("-"*100+"\n")
    print("="*100)
    print(out)
    print("-"*100)

class PaperReader:
    def __init__(self, config):
        self.paperFile = config['paper_path'] ## path to the paper
        self.databasedir = "cachedata" ## directory to store the cache files
        self.yamldb = "paper_ids.yaml" ## yaml file to store the cache file names
        self.llm = 'gpt-3.5-turbo' # large language model to use for the question answering
        self._create_model() ## create the model

    def _get_size(self, file_path):
        size = os.path.getsize(file_path)
        power = 2**10
        n = 0
        power_labels = {0: '', 1: 'KB', 2: 'MB', 3: 'GB', 4: 'TB'}
        while size > power:
            size /= power
            n += 1
        return f"{size:.2f} {power_labels[n]}"

    def _get_paper_cache_file(self):
        os.makedirs(self.databasedir, exist_ok=True)
        # Read the YAML file
        if os.path.exists(self.yamldb):
            with open(self.yamldb, 'r') as file:
                yamldata = yaml.safe_load(file)
                if yamldata is None:
                    yamldata = {}
        else:
            yamldata = {}

        if self.paperFile not in yamldata:
            yamldata[self.paperFile] = {
                'pdfdatafile': "docsearch_{}.pickle".format(str(uuid.uuid4())),
                'chaindatafile': "chain_{}.pickle".format(str(uuid.uuid4()))
                }
            with open(self.yamldb, 'w') as file:
                yaml.dump(yamldata, file)


        self.pdfdatafile1 = os.path.join(self.databasedir, yamldata[self.paperFile]['pdfdatafile'])
        self.chaindatafile1 = os.path.join(self.databasedir, yamldata[self.paperFile]['chaindatafile'])

    def _create_model(self):
        '''
        Create the model
        '''
        
        self._get_paper_cache_file()
        if config['clear_cache']:
            ## Clear the cache
            if os.path.exists(self.pdfdatafile1):
                os.remove(self.pdfdatafile1)
            
            if os.path.exists(self.chaindatafile1):
                os.remove(self.chaindatafile1)
            print("Cache cleared")

        if not os.path.exists(self.pdfdatafile1):
            # location of the pdf file/files. 
            reader = PdfReader(self.paperFile)

            # read data from the file and put them into a variable called raw_text
            raw_text = ''
            for i, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    raw_text += text


            ## Split the text into chunks of 1000 characters each with 200 characters overlap between chunks 
            ## (so that we don't miss any information) 
            text_splitter = CharacterTextSplitter(        
                separator = "\n",
                chunk_size = config['chunk_size'],
                chunk_overlap  = config['chunk_overlap'],
                length_function = len,
            )
            texts = text_splitter.split_text(raw_text)

            # if config['document_output'] and config['improve_answers']:
            #     texts = self._improve_output(texts)

            # Download embeddings from OpenAI API and create a vector store using FAISS 
            ## (a library for efficient similarity search and clustering of dense vectors)
            embeddings = OpenAIEmbeddings()

            ## Create the vector store object using FAISS 
            self.docsearch = FAISS.from_texts(texts, embeddings)

            # Save the object to a pickle file
            with open(self.pdfdatafile1, 'wb') as f:
                pickle.dump(self.docsearch, f)

            ## Print some stats
            if config['output_stats']:
                print("Number of chunks: {}".format(len(texts)))
                print("Average chunk length: {:.1f}".format(sum([len(t) for t in texts])/len(texts)))
                print("Total length of read: {} characters".format(sum([len(t) for t in texts])))
                print("Total length of original text: {} characters".format(len(raw_text)))
                print("Size of the cache pickle file: {} ".format(self._get_size(self.pdfdatafile1)))


        else:
            if config['output_stats']:
                print("Using cached data")


        # Load the object from the pickle file
        with open(self.pdfdatafile1, 'rb') as f:
            self.docsearch = pickle.load(f)

        if not os.path.exists(self.chaindatafile1):
            ## Create a question answering chain using GPT-3.5-turbo model from the langchain library 
            ## (a library for building language chains) 
            self.chain = load_qa_chain(ChatOpenAI(temperature=config['gpt_temperature'], model_name=self.llm), chain_type="stuff")
            with open(self.chaindatafile1, 'wb') as f:
                pickle.dump(self.chain, f)
        else:
            with open(self.chaindatafile1, 'rb') as f:
                self.chain = pickle.load(f)

    def _improve_output(self, texts):
        '''
        Improve the output by using the previous queries
        '''
        outdocfile = self.paperFile.replace(".pdf", "_output.md")
        
        if os.path.exists(outdocfile):
            with open(outdocfile, 'r') as f:
                text_doc = f.read()
            text_doc.replace("="*100, "")
            text_doc.replace("-"*100, "")
            total_len = len(text_doc)
            
            # print('len of text doc', len(texts_doc))
            if total_len > 1000:
                text_splitter_imp = CharacterTextSplitter(        
                    separator = "\n",
                    chunk_size = 1000,
                    chunk_overlap  = 200,
                    length_function = len,
                )
                texts_doc = text_splitter_imp.split_text(text_doc)
                texts = texts + texts_doc
                if config['output_stats']:
                    print("Using the previous queries to improve the answers")
        return texts

    def update_chain_cache(self):
        '''
        Update the chain cache
        '''
        with open(self.chaindatafile1, 'wb') as f:
            pickle.dump(self.chain, f)

    def query_document(self, query):
        '''
        Query the document and return the answer
        '''
        ## Query the document 
        instructions = " Only output the answer to the question from the article. "
        docs = self.docsearch.similarity_search(query + instructions)
        output = self.chain.run(input_documents=docs, question=query)
        self.update_chain_cache()
        return output

if __name__ == "__main__":
    main()