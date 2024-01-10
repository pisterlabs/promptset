#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import http.server
import json
import urllib.parse
import requests

OPENAI_API_KEY = "Your Api key"
OPENAI_API_ENDPOINT = "https://api.openai.com/v1/engines/davinci/completions"

class MyHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b'''
            <html>
            <body>
            <form method="POST">
                <label for="prompt">Enter your question:</label><br>
                <input type="text" id="prompt" name="prompt"><br>
                <input type="submit" value="Submit">
            </form>
            </body>
            </html>
        ''')

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        post_data = urllib.parse.parse_qs(post_data.decode('utf-8'))

        if 'prompt' in post_data:
            prompt = post_data['prompt'][0]
            
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json"
            }

            payload = {
                "prompt": prompt,
                "max_tokens": 50  # Adjust as needed
            }

            response = requests.post(OPENAI_API_ENDPOINT, json=payload, headers=headers)
            
            if response.status_code == 200:
                response_data = response.json()
                if 'choices' in response_data:
                    answer = response_data['choices'][0]['text'].strip()
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    self.wfile.write(f"<html><body><p><strong>Question:</strong> {prompt}</p><p><strong>Answer:</strong> {answer}</p></body></html>".encode())
                else:
                    self.send_response(500)
                    self.end_headers()
                    self.wfile.write(b"Error: Unable to extract valid response from OpenAI")
            else:
                self.send_response(response.status_code)
                self.end_headers()
                self.wfile.write(response.content)
        else:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(b"Bad Request: 'prompt' parameter missing")

if __name__ == '__main__':
    host = 'localhost'
    port = 8000

    with http.server.HTTPServer((host, port), MyHandler) as server:
        print(f"Server started on http://{host}:{port}")
        server.serve_forever()


# In[ ]:




