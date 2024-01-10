import os
from typing import Dict, List
import arxiv
from langchain.text_splitter import CharacterTextSplitter
from paper_class import Paper
from document_source import DocumentSource


class PaperCollection(object):
    def __init__(self,
                 openai_api_key: str,
                 chunk_size: int = 2000,
                 create_embedding: bool = True):
        """
        Initialize a PaperCollection instance.

        Args:
            openai_api_key (str): The API key for OpenAI.
            chunk_size (int, optional): The size (in characters) for splitting text chunks. Defaults to 2000.
            create_embedding (bool): Whether to create embeddings while adding the papers.
        """
        self.create_embedding_ = create_embedding
        self.papers: Dict[str, Paper] = {}
        self.document_source_: DocumentSource = DocumentSource(openai_api_key)
        self.text_splitter_: CharacterTextSplitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)

    def get_paper(self, title: str) -> Paper:
        """
        Get a paper from the collection.

        Args:
            title (str): The title of requested paper.

        Returns:
            The Paper object with the requested title.

        Raises:
            KeyError: when the paper of the given title not existed in this collection.
        """
        return self.papers[title]

    def add_paper(self, paper: Paper):
        """
        Add a Paper object to the collection.

        Args:
            paper (Paper): The paper to add.
        """
        print(f"Adding paper {paper.title} to the collection...")
        if paper.title in self.papers:
            print(f"Paper {paper.title} already exists in the collection.")
            return

        self.papers[paper.title] = paper
        if not self.create_embedding_:
            return
        docs = self.text_splitter_.create_documents([f'Title: {paper.title}\nAbstract: {paper.summary}'])
        for doc in docs:
            doc.metadata['source'] = paper.title
        self.document_source_.add_documents(docs)

    def add_paper_dict(self, paper_dict: Dict[str, Paper]):
        """
        Add multiple papers to the collection from a dictionary.

        Args:
            paper_dict (Dict[str, Paper]): A dictionary of papers to add.
        """
        self.papers.update(paper_dict)
        for _, paper in paper_dict.items():
            self.add_paper(paper)

    def add_from_arxiv(self,
                       search,
                       download: bool = False):
        """
        Search for papers on arXiv and optionally download them.

        Args:
            search: A container of arXiv search results.
            download (bool, optional): Whether to download the papers. Defaults to False.
        """
        output_directory: str = "arxiv_papers"
        os.makedirs(output_directory, exist_ok=True)

        for result in search.results():
            paper: Paper = Paper(
                title=result.title,
                summary=result.summary,
                url=result.pdf_url,
                authors=[author.name for author in result.authors],
                publish_date=result.published,
                on_arxiv=True,
            )
            self.add_paper(paper)
            if download:
                paper.download(use_title=True)

    def latex_bibliography(self) -> list:
        return [paper.get_latex_citation() for title, paper in self.papers.items()]

    def query_papers(self, **kwargs) -> dict[str, Paper]:
        """
        Retrieve papers related to a specific queried topic.

        Args:
            **kwargs (dict): The args used by retrieve function of DocumentSource class.

        Returns:
            Dict[str, Paper]: A dictionary of papers related to the topic.
        """
        print(f"Sourcing the papers related to with query {str(kwargs)}...")
        if len(self.papers) == 0:
            raise ValueError("The paper collection is empty.")

        source_documents = self.document_source_.retrieve(**kwargs)

        paper_dict = {}
        for doc, score in source_documents:
            title = doc.metadata['source']
            if title not in paper_dict:
                print(f"Found paper with score {score:.2f}: {title};")
                paper_dict[title] = self.papers[title]

        return paper_dict

if __name__ == '__main__':
    from test_utils import get_test_papers

    search = arxiv.Search(
        query = "au:Yanrui Du AND ti:LLM",
        max_results = 3,
        sort_by = arxiv.SortCriterion.SubmittedDate,
        sort_order = arxiv.SortOrder.Descending
    )

    paper_collection = PaperCollection(
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        chunk_size=1000,
    )
    
    paper_collection.add_from_arxiv(
        search,
        download=False,
    )
    
    paper_collection.add_paper_dict(get_test_papers())

    print('\n\n'.join(paper_collection.latex_bibliography()))

    for title, paper in paper_collection.papers.items():
        print(paper.get_arxiv_citation())
        print(paper.get_APA_citation())
        print(paper.get_latex_citation())
        
    papers = paper_collection.query_papers(
        query="CALLA Dataset",
        num_retrieval=1,
    )
    
    for title, paper in papers.items():
        print(paper.get_arxiv_citation())
        print(paper.get_APA_citation())
        print(paper.get_latex_citation())

