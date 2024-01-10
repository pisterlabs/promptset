from langchain.chains import RetrievalQAWithSourcesChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.memory.utils import get_prompt_input_key
from langchain.vectorstores.base import VectorStoreRetriever, VectorStore
from config import cfg
from typing import Any, Dict, Tuple, List, TypeVar


import os
from pathlib import Path

from generate_embeddings import load_pdfs, generate_embeddings
from hr_model import QAData
from log_init import logger

VST = TypeVar("VST", bound="VectorStore")


class KeySourceMemory(ConversationSummaryBufferMemory):
    def _get_input_output(
        self, inputs: Dict[str, Any], outputs: Dict[str, str]
    ) -> Tuple[str, str]:
        if self.input_key is None:
            prompt_input_key = get_prompt_input_key(inputs, self.memory_variables)
        else:
            prompt_input_key = self.input_key
        if self.output_key is None:
            output_key = "answer"
        else:
            output_key = self.output_key
        return inputs[prompt_input_key], outputs[output_key]


def load_embeddinges(
    embedding_dir: Path = cfg.faiss_persist_directory_uk,
    doc_location: Path = cfg.doc_location_uk,
) -> Tuple[VST, List[Document]]:
    """
    Loads the PDF documents to support text extraction in the Chainlit UI.
    In case there are no persisted embeddings, the embeddings are generated.
    In case the embeddings are persisted, then they are loaded from the file system.

    Returns:
    Tuple[VST, List[Document]]: Recturs a reference to the vector store and the list of all pdf files.
    """
    logger.info(f"Checking: {embedding_dir}")
    documents = load_pdfs(doc_location)
    assert len(documents) > 0
    if embedding_dir.exists() and len(list(embedding_dir.glob("*"))) > 0:
        logger.info(f"reading from existing directory: {embedding_dir}")
        docsearch = FAISS.load_local(embedding_dir, cfg.embeddings)
        return docsearch, documents
    return (
        generate_embeddings(documents, doc_location, embedding_dir.absolute()),
        documents,
    )


template = """Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES"). If you know a joke about the subject, make sure that you include it in the response.
If you don't know the answer, say that you don't know and make up some joke about the subject. Don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.

QUESTION: Which state/country's law governs the interpretation of the contract?
=========
Content: This Agreement is governed by English law and the parties submit to the exclusive jurisdiction of the English courts in  relation to any dispute (contractual or non-contractual) concerning this Agreement save that either party may apply to any court for an  injunction or other relief to protect its Intellectual Property Rights.
Source: 28-pl
Content: No Waiver. Failure or delay in exercising any right or remedy under this Agreement shall not constitute a waiver of such (or any other)  right or remedy.\n\n11.7 Severability. The invalidity, illegality or unenforceability of any term (or part of a term) of this Agreement shall not affect the continuation  in force of the remainder of the term (if any) and this Agreement.\n\n11.8 No Agency. Except as expressly stated otherwise, nothing in this Agreement shall create an agency, partnership or joint venture of any  kind between the parties.\n\n11.9 No Third-Party Beneficiaries.
Source: 30-pl
Content: (b) if Google believes, in good faith, that the Distributor has violated or caused Google to violate any Anti-Bribery Laws (as  defined in Clause 8.5) or that such a violation is reasonably likely to occur,
Source: 4-pl
=========
FINAL ANSWER: This Agreement is governed by English law.
SOURCES: 28-pl

QUESTION: What did the president say about Michael Jackson?
=========
Content: Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans.  \n\nLast year COVID-19 kept us apart. This year we are finally together again. \n\nTonight, we meet as Democrats Republicans and Independents. But most importantly as Americans. \n\nWith a duty to one another to the American people to the Constitution. \n\nAnd with an unwavering resolve that freedom will always triumph over tyranny. \n\nSix days ago, Russia’s Vladimir Putin sought to shake the foundations of the free world thinking he could make it bend to his menacing ways. But he badly miscalculated. \n\nHe thought he could roll into Ukraine and the world would roll over. Instead he met a wall of strength he never imagined. \n\nHe met the Ukrainian people. \n\nFrom President Zelenskyy to every Ukrainian, their fearlessness, their courage, their determination, inspires the world. \n\nGroups of citizens blocking tanks with their bodies. Everyone from students to retirees teachers turned soldiers defending their homeland.
Source: 0-pl
Content: And we won’t stop. \n\nWe have lost so much to COVID-19. Time with one another. And worst of all, so much loss of life. \n\nLet’s use this moment to reset. Let’s stop looking at COVID-19 as a partisan dividing line and see it for what it is: A God-awful disease.  \n\nLet’s stop seeing each other as enemies, and start seeing each other for who we really are: Fellow Americans.  \n\nWe can’t change how divided we’ve been. But we can change how we move forward—on COVID-19 and other issues we must face together. \n\nI recently visited the New York City Police Department days after the funerals of Officer Wilbert Mora and his partner, Officer Jason Rivera. \n\nThey were responding to a 9-1-1 call when a man shot and killed them with a stolen gun. \n\nOfficer Mora was 27 years old. \n\nOfficer Rivera was 22. \n\nBoth Dominican Americans who’d grown up on the same streets they later chose to patrol as police officers. \n\nI spoke with their families and told them that we are forever in debt for their sacrifice, and we will carry on their mission to restore the trust and safety every community deserves.
Source: 24-pl
Content: And a proud Ukrainian people, who have known 30 years  of independence, have repeatedly shown that they will not tolerate anyone who tries to take their country backwards.  \n\nTo all Americans, I will be honest with you, as I’ve always promised. A Russian dictator, invading a foreign country, has costs around the world. \n\nAnd I’m taking robust action to make sure the pain of our sanctions  is targeted at Russia’s economy. And I will use every tool at our disposal to protect American businesses and consumers. \n\nTonight, I can announce that the United States has worked with 30 other countries to release 60 Million barrels of oil from reserves around the world.  \n\nAmerica will lead that effort, releasing 30 Million barrels from our own Strategic Petroleum Reserve. And we stand ready to do more if necessary, unified with our allies.  \n\nThese steps will help blunt gas prices here at home. And I know the news about what’s happening can seem alarming. \n\nBut I want you to know that we are going to be okay.
Source: 5-pl
Content: More support for patients and families. \n\nTo get there, I call on Congress to fund ARPA-H, the Advanced Research Projects Agency for Health. \n\nIt’s based on DARPA—the Defense Department project that led to the Internet, GPS, and so much more.  \n\nARPA-H will have a singular purpose—to drive breakthroughs in cancer, Alzheimer’s, diabetes, and more. \n\nA unity agenda for the nation. \n\nWe can do this. \n\nMy fellow Americans—tonight , we have gathered in a sacred space—the citadel of our democracy. \n\nIn this Capitol, generation after generation, Americans have debated great questions amid great strife, and have done great things. \n\nWe have fought for freedom, expanded liberty, defeated totalitarianism and terror. \n\nAnd built the strongest, freest, and most prosperous nation the world has ever known. \n\nNow is the hour. \n\nOur moment of responsibility. \n\nOur test of resolve and conscience, of history itself. \n\nIt is in this moment that our character is formed. Our purpose is found. Our future is forged. \n\nWell I know this nation.
Source: 34-pl
=========
FINAL ANSWER: The president did not mention Michael Jackson. And here is a joke about Michael Jackson: Why did Michael Jackson go to the bakery? Because he wanted to "beat it" and grab some "moon-pies"! 
SOURCES:

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:"""
HUMOUR_PROMPT = PromptTemplate(
    template=template, input_variables=["summaries", "question"]
)


def create_retrieval_chain(
    docsearch: VST, verbose: bool = False, humour: bool = True
) -> RetrievalQAWithSourcesChain:
    """
    This function creates the QA chain with memory and in case the humour parameter is true,
    then a manipulated prompt - that tends to create jokes on certain occasions - is used.

    Parameters:
    docsearch (VST): A reference to the vector store.
    verbose (bool): Determines whether LangChain's internal logging is printed to the console or not.
    humour (bool): Determines whether the prompt for answers with jokes is used or not.

    Returns:
    RetrievalQAWithSourcesChain: The QA chain
    """
    memory = KeySourceMemory(llm=cfg.llm, input_key="question", output_key="answer")
    chain_type_kwargs = {}
    if verbose:
        chain_type_kwargs["verbose"] = True
    if humour:
        chain_type_kwargs["prompt"] = HUMOUR_PROMPT
    search_retriever: VectorStoreRetriever = docsearch.as_retriever()
    search_retriever.search_kwargs = {"k": cfg.search_results}
    qa_chain = RetrievalQAWithSourcesChain.from_chain_type(
        cfg.llm,
        retriever=search_retriever,
        chain_type="stuff",
        memory=memory,
        chain_type_kwargs=chain_type_kwargs,
    )

    return qa_chain


def load_all_chains(country_filter: str = None) -> Dict[str, QAData]:
    res = {}
    for country, v in cfg.location_persistence_map.items():
        if country_filter is None or country_filter == country:
            faiss_persist_directory = v["faiss_persist_directory"]
            doc_location = v["doc_location"]
            vst, documents = load_embeddinges(faiss_persist_directory, doc_location)
            chain = create_retrieval_chain(vst, humour=os.getenv("HUMOUR") == "true")
            res[country] = QAData(vst=vst, documents=documents, chain=chain)
    return res


if __name__ == "__main__":
    chain_dict = load_all_chains()
    logger.info(len(chain_dict.items()))
