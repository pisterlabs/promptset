import requests
import logging
import json
import tiktoken
import gradio as gr
from typing import Any, List
from langchain.schema import Document
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from utils import json_validator, fetch_chat


class LatexTextSplitter(RecursiveCharacterTextSplitter):
    """Attempts to split the text along Latex-formatted layout elements."""

    def __init__(self, **kwargs: Any):
        """Initialize a LatexTextSplitter."""
        separators = [
            # First, try to split along Latex sections
            "\chapter{",
            "\section{",
            "\subsection{",
            "\subsubsection{",
            # Now split by environments
            "\begin{"
            # "\n\\begin{enumerate}",
            # "\n\\begin{itemize}",
            # "\n\\begin{description}",
            # "\n\\begin{list}",
            # "\n\\begin{quote}",
            # "\n\\begin{quotation}",
            # "\n\\begin{verse}",
            # "\n\\begin{verbatim}",
            ## Now split by math environments
            # "\n\\begin{align}",
            # "$$",
            # "$",
            # Now split by the normal type of lines
            " ",
            "",
        ]
        super().__init__(separators=separators, **kwargs)


class Suggest:
    def __init__(self, max_ideas: int, model: str = "gpt-3.5-turbo"):
        self.max_ideas = max_ideas
        self.encoder = tiktoken.encoding_for_model(model)
        self.model = model
        self.idea_list = []
        with open("./sample/sample.tex", "r") as f:
            self.sample_content = f.read()

    def split_chunk(
        self, latex_whole_document: str, chunk_size: int = 2000, retry: int = 5
    ) -> List[Document]:
        chunk_size = min(chunk_size, len(latex_whole_document))

        for _ in range(retry):
            try:
                latex_splitter = LatexTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=0,
                )
                docs = latex_splitter.create_documents([latex_whole_document])
                return docs
            except:
                chunk_size = chunk_size // 2

        raise Exception("Latex document split check failed.")

    def analyze(
        self, latex_whole_document: str, openai_key: str, progress: gr.Progress
    ):
        logging.info("start analysis")
        docs = self.split_chunk(latex_whole_document)
        progress(0.05)

        output_format = """

        ```json
        [
            \\ Potential point for improvement 1
            {{
                "title": string \\ What this modification is about
                "thought": string \\ The reason why this should be improved
                "action": string \\ how to make improvement
                "original": string \\ the original latex snippet that can be improved
                "improved": string \\ the improved latex snippet which address your point
            }},
            {{}}
        ]
        ```
        """

        ideas = []
        for doc in progress.tqdm(docs):
            prompt = f"""
            I'm a computer science student.
            You are my editor.
            Your goal is to improve my paper quality at your best.


            ```
            {doc.page_content}
            ```
            The above is a segment of my research paper. If the end of the segment is not complete, just ignore it.
            Point out the parts that can be improved.
            Focus on grammar, writing, content, section structure.
            Ignore comments and those that are outside the document environment.
            List out all the points with a latex snippet which is the improved version addressing your point.
            Same paragraph should be only address once.
            Output the response in the following valid json format:
            {output_format}

            """

            idea = fetch_chat(prompt, openai_key, model=self.model)
            idea = json_validator(idea, openai_key)
            if isinstance(idea, list):
                ideas += idea
                if len(ideas) >= self.max_ideas:
                    break
            else:
                # raise gr.Error(idea)
                continue

        if not ideas:
            raise gr.Error("No suggestions generated.")

        logging.info("complete analysis")
        return ideas

    def read_file(self, f: str):
        if f is None:
            return ""
        elif f.name.endswith("pdf"):
            loader = UnstructuredPDFLoader(f.name)
            pages = loader.load_and_split()
            return "\n".join([p.page_content for p in pages])
        elif f.name.endswith("tex"):
            with open(f.name, "r") as f:
                return f.read()
        else:
            return "Only support .tex & .pdf"

    def generate(self, txt: str, openai_key: str, progress=gr.Progress()):
        if not openai_key:
            raise gr.Error("Please provide openai key !")

        try:
            idea_list = self.analyze(txt, openai_key, progress)
            self.idea_list = idea_list
            k = min(len(idea_list), self.max_ideas)

            idea_buttons = [
                gr.Button.update(visible=True, value=i["title"])
                for e, i in enumerate(idea_list[: self.max_ideas])
            ]
            idea_buttons += [gr.Button.update(visible=False)] * (
                self.max_ideas - len(idea_buttons)
            )

            idea_details = [
                gr.Textbox.update(value="", label="thought", visible=True),
                gr.Textbox.update(value="", label="action", visible=True),
                gr.Textbox.update(
                    value="", label="original", visible=True, max_lines=5, lines=5
                ),
                gr.Textbox.update(
                    value="", label="improved", visible=True, max_lines=5, lines=5
                ),
            ]

            return (
                [
                    gr.Textbox.update(
                        "Suggestions", interactive=False, show_label=False
                    ),
                    gr.Button.update(visible=True, value="Analyze"),
                ]
                + idea_details
                + idea_buttons
            )
        except Exception as e:
            raise gr.Error(str(e))
