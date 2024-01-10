import threading
from pymed import PubMed
from cat.mad_hatter.decorators import tool, hook
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain


class SchrodingerCat:

    def __init__(self, ccat):

        # PyMed
        self.pymed = PubMed(tool="mytool", email="myemail@email.com")

        # cheshire_cat
        self.cat = ccat

    @staticmethod
    def parse_query(tool_input):

        # Split the inputs
        multi_input = tool_input.split(",")

        # Cast max_results to int
        try:
            max_results = int(multi_input[1])
        except ValueError:
            # If the model leave a quote remove it
            max_results = int(multi_input[1].strip("'"))

        # Query for PubMed
        query = f"{multi_input[0]}[Title]"

        return query, max_results

    @staticmethod
    def parse_results(results):
        cleaned = []

        # Loop all results
        for result in results:

            # Make Dict
            r = result.toDict()

            # Drop useless keys
            r.pop("xml")
            r.pop("pubmed_id")

            cleaned.append(r)

        return cleaned

    def __query(self, query: str, max_results: int = 1):

        # Query PubMed
        results = self.pymed.query(query=query, max_results=max_results)

        # Store docs in Working Memory for further operations.
        # e.g. filter docs
        self.cat.working_memory["pubmed_results"] = self.parse_results(results)

    def make_search(self, tool_input):
        # Split input in str and int
        query, max_results = self.parse_query(tool_input)

        # Make concurrent task to download paper in background if max_results is high
        search = threading.Thread(target=self.__query, name="PubMed query", args=[query, max_results])
        search.start()


@tool(return_direct=True)
def simple_search(query: str, cat):
    """
    Useful to look for a query on PubMed. It is possible to specify the number of results desired.
    The input to this tool should be a comma separated list of a string and an integer number.
    The integer number is optional and if not provided is set to 1.
    For example: 'Antibiotic,5' would be the input if you want to look for 'Antibiotic' with max 5 results.
    Another example: 'Antibiotic,1' would be the input if only the query 'Antibiotic' is asked.
    To use this tool start the whole prompt with PUBMED: written in uppercase.
    Examples:
         - PUBMED: Look for "Public Healthcare" and give me 3 results. Input is 'Public Healthcare,3'
         - PUBMED: Look for "Antibiotic resistance". Input is 'Public Healthcare,1'
    """

    # Schrodinger Cat
    schrodinger_cat = SchrodingerCat(cat)

    # Search on PubMed
    schrodinger_cat.make_search(query)

    # TODO: change this output
    out = f"Alright. I'm looking for {schrodinger_cat.parse_query(query)[1]} results about" \
          f" {schrodinger_cat.parse_query(query)[0].strip('[Title]')} on PubMed. This may take some time. " \
          f"Hang on please, I'll tell you when I'm done"

    return out


@tool()
def empty_working_memory(tool_input, cat):
    """
    Useful to empty and forget all the documents in the Working Memory. Input is always None.
    """
    if "pubmed_results" in cat.working_memory:
        cat.working_memory.pop("pubmed_results")

    # TODO: this has to be tested
    # the idea is having the Cat answer without directly returning a hard coded output string
    return cat.llm("Can you forget everything I asked you to keep in mind?")


@tool(return_direct=True)
def summary_working_memory(tool_input, cat):
    """
    Useful to ask for a detailed summary of what's in the Cat's Working Memory. Input is always None.
    Example:
        - What's in your memory? -> use summary_working_memory tool
        - Tell me the papers you have currently in memory
    """
    # Memories in Working Memory
    if "pubmed_results" in cat.working_memory.keys():
        memories = cat.working_memory["pubmed_results"]

        n_memories = len(memories)

    else:
        memories = []
        n_memories = 0

    if n_memories == 0:
        return cat.llm("Say that you memory is empty")

    prefix = f"Currently I have {n_memories} papers temporarily loaded in memory.\n"
    papers = ""
    for m in memories:
        papers += f"{m}\n"

    return prefix + papers


@tool
def include_paper(tool_input, cat):
    """
    Useful to ask a question about the abstracts retrieved and saved in the working memory.
    The Input is a question wrapped in double quotes.
    """

    # Get abstracts
    if "pubmed_results" in cat.working_memory.keys():
        abstracts = [f"Title: {m['title']}\nAbstract: {m['abstract']}" for m in cat.working_memory["pubmed_results"]]

        abstracts = [Document(page_content=a) for a in abstracts]

        chain = load_qa_chain(cat.llm, chain_type="refine")
        answer = chain.run(input_documents=abstracts, question=tool_input)

    return answer


