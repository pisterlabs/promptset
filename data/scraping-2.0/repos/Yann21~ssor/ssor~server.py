import urllib.parse
import re
from config import persist_directory, openai_api_key
from http.server import BaseHTTPRequestHandler, HTTPServer
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()

        # Input
        request_str = urllib.parse.unquote(self.path.split("/api/")[-1])

        # Retrieve docs
        search_results = vectordb.similarity_search_with_score(request_str, k=15)
        retrieved_docs = sorted(search_results, key=lambda x: x[1], reverse=True)
        org_link_format = "[%.2f]: [[id:%s][%s]] \n %s"
        docs = [org_link_format % (score, doc.metadata["ID"],
                                   doc.metadata["title"].strip(),
                                   doc.metadata["hierarchy"].strip())
                for doc, score in retrieved_docs]

        # Format the output
        response_str = f"#+title: Most similar nodes \n\n:QUERY:\n{request_str} \n:END:\n\n"
        for i, source in enumerate(docs):
            response_str += "* " + source + "\n"

        self.wfile.write(response_str.encode())

def run_server():
    server_address = ('', 8800)
    httpd = HTTPServer(server_address, RequestHandler)
    print(f'Server is running on port {server_address[1]}')
    httpd.serve_forever()

if __name__ == '__main__':
    embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vectordb = Chroma("langchain_store", embedding_function=embedding, persist_directory=persist_directory)
    run_server()
