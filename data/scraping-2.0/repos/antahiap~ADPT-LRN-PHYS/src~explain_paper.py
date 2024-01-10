from openai_api import OpenAIApi
from openai_multi_client import OpenAIMultiClient
import logging
import tiktoken

from dotenv import load_dotenv
load_dotenv()

class ExplainPaper():
    def __init__(self, paper_name, paper_json, max_chunk_size=15000):
        self.api = OpenAIApi()
        self.paper_name = paper_name
        self.paper_json = paper_json
        self.max_chunk_size = max_chunk_size
        self.explanation = ""
        self.chunk_explanations = []
        self.chunks = []
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def generate_explanation(self):
        self.make_chunks()
        self.get_paper_explanation()

    def explanation_prompt(self, context):
        return f"Answer using markdown. Explain in every details using markdown:\n{context}\n"
    
    def add_explanation(self, result):
        id = result.metadata['id']
        explanation = result.response['choices'][0]['message']['content']
        self.chunk_explanations[id] = explanation

    def get_paper_explanation(self):
        api = OpenAIMultiClient(concurrency=20, endpoint="chats", data_template={"model": "gpt-3.5-turbo-16k"})
        def make_requests():
            for index, chunk in enumerate(self.chunks):
                prompt = self.explanation_prompt(chunk)
                print(f"Request {index} {chunk[:10]}")
                api.request(
                    data={"messages": [{"role": "user", "content": prompt}]},
                    metadata={'id': index},
                )
        api.run_request_function(make_requests)

        self.chunk_explanations = ["" for x in self.chunks]
        for result in api:
            self.add_explanation(result)
        self.explanation = "\n".join(self.chunk_explanations)

    def check_new_text_length(self, new_text):
        encoded_new_text = self.encoding.encode(new_text)
        if len(encoded_new_text) > self.max_chunk_size:
            logging.warning(f"New text is too long: {new_text[:30]}")
            new_text = self.encoding.decode(encoded_new_text[:self.max_chunk_size])
        return new_text

    def check_new_chunk(self, text, new_text):
        new_text = self.check_new_text_length(new_text)
        if len(self.encoding.encode(text + new_text)) > self.max_chunk_size:
            self.chunks.append(text)
            text = new_text
        else:
            text += new_text
        return text
        
    def get_subsection_text(self, subsection, text="", level=1):
        for s in subsection:
            if s.get('id'):
                new_text = f"{'#' * level} {s['id']} {s['section']}\n{s['text']}\n"
                text = self.check_new_chunk(text, new_text)
                text = self.get_subsection_text(s['subsection'], text, level=level+1)
        return text

    def make_chunks(self):
        text = self.get_subsection_text(self.paper_json)
        self.chunks.append(text)
    
if __name__ == "__main__":
    import json
    with open("./data/article_pdf/txt/2309.03409.json", "r") as f:
        paper_json = json.load(f)
    explain_paper = ExplainPaper("2309.03409", paper_json, 15000)
    explain_paper.generate_explanation()
    for chunk in explain_paper.chunks:
        print(len(explain_paper.encoding.encode(chunk)))
    print("N chunks: ", len(explain_paper.chunks))
    print("Explanation: ", explain_paper.explanation)