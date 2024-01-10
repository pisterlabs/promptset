import http.server
import json
import os
from http.server import BaseHTTPRequestHandler

# Import your existing code
import constants
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

# Set OPENAI_API_KEY
os.environ["OPENAI_API_KEY"] = constants.API_KEY

# Create a TextLoader and an Index
loader1 = TextLoader('data.txt')
index = VectorstoreIndexCreator().from_loaders([loader1])

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/query":
            # Handle GET request for the "/query" route if needed
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write("GET request received for /query".encode("utf-8"))
        else:
            self.send_response(404)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write("Not Found".encode("utf-8"))

    def do_POST(self):
        if self.path == "/query":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode("utf-8"))

            # Get the user's query from the request data
            query = data.get("query")

            # Query the index
            llm = ChatOpenAI()
            result = index.query(query, llm=llm)

            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Access-Control-Allow-Origin", "http://127.0.0.1:5500")  # Allow requests from your frontend origin
            self.end_headers()

            # Send the result back to the client
            self.wfile.write(result.encode("utf-8"))
        else:
            self.send_response(404)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write("Not Found".encode("utf-8"))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "http://127.0.0.1:5500")  # Replace with your frontend's origin
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

if __name__ == "__main__":
    server_address = ("", 8000)
    httpd = http.server.HTTPServer(server_address, RequestHandler)
    print("Server running on port 8000...")
    httpd.serve_forever()
