import PyPDF2
from dotenv import load_dotenv
from langchain.agents import AgentType, initialize_agent
from langchain.llms import OpenAI
from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper

# Load environment variables
load_dotenv()


class ColdMailing:
    """
    Generates a cold email template.

    Parameters
    ----------
    company_name : str
    name : str
    email : str
    resume : file
    """

    def __init__(self) -> None:
        self.llm = OpenAI(temperature=0.6, max_tokens=1000)

        search = GoogleSearchAPIWrapper(k=1)

        self.tools = [
            Tool(
                name="Google Search",
                description="Search Google for results about the company or professor. Use it only once in any case",
                func=search.run,
            )
        ]

        self.agent = initialize_agent(
            self.tools, self.llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True)

    def generate(self, target_name, name, resume, target_type, requested_position):
        """
        Generates a cold email template.
        
        Parameters
        ----------
        target_name : str
        name : str
        resume : file
        target_type : str
        requested_position : str
        """

        return self.agent.run(f"""
            Write a cold mail to {target_type} called {target_name} for the position of {requested_position} with following conditons:
            if it is a professor then use his/her top 3 best research paper work by using google scholar else if it is a company then use my info to show my passion.

            Use the following information to create the cold mail:
            Name: {name}
            {resume}
            """)


class PDFTextExtractor:
    """
    Extracts text from a PDF file.

    Parameters
    ----------
    file : file
        The PDF file to extract text from.

        Returns
        -------
        text : str
    """

    def extract(self, file):
        """
        Extracts text from a PDF file.
        
        Parameters
        ----------
        file : file
        """
        text = str()
        file_reader = PyPDF2.PdfReader(file)
        for page in file_reader.pages:
            text += page.extract_text()

        return text
