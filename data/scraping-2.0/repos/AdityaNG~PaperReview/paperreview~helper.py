import openai
import requests
import tiktoken
from pypdf import PdfReader
from tqdm import tqdm

from .constants import OPENAI_API_KEY


def download_pdf_from_url(url: str):
    """
    Download a PDF file from a URL and save it to the current directory.
    """
    response = requests.get(url, stream=True)
    filename = url.split("/")[-1]
    if not filename.endswith(".pdf"):
        filename += ".pdf"

    with open(filename, "wb") as pdf:
        for chunk in tqdm(response.iter_content(chunk_size=1024)):
            if chunk:
                pdf.write(chunk)

    return filename


def extract_text_from_pdf(pdf_path: str):
    """
    Extract text from a PDF file.
    """
    pdf_file_obj = open(pdf_path, "rb")
    pdf_reader = PdfReader(pdf_file_obj)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    pdf_file_obj.close()
    return text


class ChunkIterator:
    def __init__(self, text, n, model):
        self.text = text
        self.n = n
        self.model = model
        self.encoding = tiktoken.encoding_for_model(model)
        self.ub = 0
        self.lb = 0
        self.num_chunks = self.calculate_num_chunks()

    @staticmethod
    def num_tokens_from_string(
        string: str, encoding: tiktoken.core.Encoding
    ) -> int:
        """Returns the number of tokens in a text string."""
        num_tokens = len(encoding.encode(string))
        return num_tokens

    def calculate_num_chunks(self):
        count = 0
        for _ in self:
            count += 1
        self.lb = 0
        self.ub = 0
        return count

    def __iter__(self):
        return self

    def __next__(self):
        if self.ub >= len(self.text):
            raise StopIteration
        while self.num_tokens_from_string(
            self.text[self.lb : self.ub], self.encoding  # noqa
        ) < self.n and self.ub < len(self.text):
            self.ub += 1
        chunk = self.text[self.lb : self.ub]  # noqa
        self.lb = self.ub
        return chunk

    def __len__(self):
        return self.num_chunks


def ask_paper_text(
    text: str,
    question: str,
    api_key: str = OPENAI_API_KEY,
    model: str = "text-davinci-003",
    max_tokens: int = 1000,
    temperature: float = 0.5,
):  # pragma: no cover
    """
    Ask a question about a research paper using GPT.
    """
    openai.api_key = api_key

    summary = ""
    for text_segment in tqdm(
        ChunkIterator(text, 3000, model),
        desc="Reading paper",
    ):
        prompt = f"Following is a segment of a Research Paper. \
            Answer the question that follows. \n\n{text_segment} \
            \n\n{question} ; Only use bullet points. \
            Be concise and use academic language"
        response_gpt = openai.Completion.create(
            model=model,
            max_tokens=max_tokens,
            prompt=prompt,
            temperature=temperature,
        )
        summary += response_gpt.choices[0].text.strip()

    prompt = f"Following is a summary of a Research Paper. \
        Answer the question that follows. \n\n{text_segment} \
        \n\n{question} ; Only use bullet points. \
        Be concise and use academic language"
    response_gpt = openai.Completion.create(
        model=model,
        max_tokens=max_tokens,
        prompt=prompt,
        temperature=temperature,
    )
    answer = response_gpt.choices[0].text.strip()

    return answer
