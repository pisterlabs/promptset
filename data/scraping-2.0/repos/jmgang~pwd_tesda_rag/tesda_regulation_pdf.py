from typing import List, Optional
from langchain.docstore.document import Document
from langchain.load.serializable import Serializable


class TesdaRegulationPDF(Serializable):
    name: str = ''
    documents: List[Document]
    toc_page: Optional[int]
    core_pages: List[int] = []
    competency_map_pages: List[int] = []
    trainee_entry_requirements_pages: List[int] = []
    section1_pages: List[int] = []
    cc_short_pages: List[int] = []
    cleaned: Optional[dict]
    summary: Optional[dict]

    def __repr__(self) -> str:
        return (f"TesdaRegulationPDF(Name:{self.name}, Length: {len(self.documents)} documents, TOC page: {self.toc_page}, "
                f"Core pages: {self.core_pages}, Competency Map page: {self.competency_map_pages}, "
                f"Trainee Entry Requirements pages: {self.trainee_entry_requirements_pages}), "
                f"Section 1 pages: {self.section1_pages}, CC Short pages: {self.cc_short_pages}")
