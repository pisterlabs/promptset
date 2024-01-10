from database import Neo4jDatabase
from pydantic import BaseModel, Extra
from langchain.prompts.base import BasePromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.base import Chain
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.vectorstores import Chroma
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.chat_models import ChatOpenAI
from typing import Any, Dict, List
from logger import logger

example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)

# These are a lot of examples of a pretend task of creating antonyms.
examples = [
    {"input": "What causes glioblastoma", "output": "which genes is associated with glioblastoma"},
    {"input": "what genes regulate AXIN1", "output": "what genes upregulate and downregulate AXIN1"},
    {"input": "Is POLD4 a drug target? ", "output": "Which drug affect POLD4?"},
  
]

# Form a prompt that instructs the model to generate questions based on the input.
        
# prompt = """You are a expert in drug discovery.
#         Your task is to provide the user more detailed questions to query in a knowledge graph database if their questions \n
#         are too general or irrelevant to information in the knowledge graph. 
#         The knowledge graph databases contains biological processes, cellular component, disease\n
#         drug, gene protein, moleular funciton, pathway ant tissue as nodes and edges that describe their relationship.
#         Some examples are:""" + examples+ """. You don't need to apologize. 
#         Make your question as simple as possible"""

class QuestionGenerationTool(Chain):

    # graph: Neo4jDatabase = Field(exclude=True)
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:
    embeddings: OpenAIEmbeddings = OpenAIEmbeddings()

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys.
        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys.
        :meta private:
        """
        _output_keys = [self.output_key]
        return _output_keys


    def _call(self, inputs: Dict[str, str]) -> Dict[str, Any]:
        """Embed a question and do vector search."""
        question = inputs[self.input_key]
        logger.debug(f"Vector embed question input: {question}")
        embedding = self.embeddings.embed_query(question)

        example_selector = SemanticSimilarityExampleSelector.from_examples(
                                examples, OpenAIEmbeddings(), Chroma, k=1)

        self.callback_manager.on_text(
            "Vector search embeddings:", end="\n", verbose=self.verbose
        )
        self.callback_manager.on_text(
            embedding[:5], color="green", end="\n", verbose=self.verbose
        )

        '''add list of example question embedding'''
        similar_prompt = FewShotPromptTemplate(
                                    # We provide an ExampleSelector instead of examples.
                                    example_selector=example_selector,
                                    example_prompt=example_prompt,
                                    prefix="Give a similar question to user input which suits better for cypher query",
                                    suffix="Input: {question}\nOutput:", 
                                    input_variables=["question"],)
        
        chain = LLMChain(llm = ChatOpenAI(temperature=0), prompt = similar_prompt)
        context = chain.run(self.input_key)
        return {self.output_key: context}
        
        

    
    def __init__(self, llm, verbose=False):
        self.llm = llm
        self.verbose = verbose



