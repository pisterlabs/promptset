from langchain.prompts.example_selector import (
    MaxMarginalRelevanceExampleSelector,
)
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate, FewShotPromptTemplate


class DittoImageRAGExampleStore:
    def __init__(self):
        # example template will be used to match query / example pairs.
        self.example_template = PromptTemplate(
            input_variables=["query", "response"],
            template="user's query: {query}\nresponse:\n{response}",
        )
        self.construct()

    def construct(self):
        self.create_query_example_pairs()
        self.create_example_store()

    def create_query_example_pairs(self):
        """creates query / example pairs and saves to self.example_pairs"""

        self.example_pairs = [
            {
                "query": "can you tell me a description of the image?\nimage's caption: a picture of a car",
                "response": "<QA> What is the color of the car?\n<QA Response> red\n<QA> What is the make of the car?\n<QA Response> Tesla\n<QA> Where is the car parked?\n<QA Response> driveway\n<DONE> I see a red car that is parked in the driveway. The car looks like a Tesla.",
            },
            {
                "query": "How many fingers am I holding up?",
                "response": "<QA> How many fingers are there?\n<QA Response> 5\n<DONE> I see 5 fingers.",
            },
            {
                "query": "can you tell me a description of the image?\nimage's caption: man has brown hair",
                "response": "<QA> What is the man doing?\n<QA Response> sitting\n<QA> What color is the man's eyes?\n<QA Response> blue\n<DONE> I see a man with brown hair and blue eyes sitting.",
            },
            {
                "query": "What do you think about this shirt? Does it go with my hair?",
                "response": "<QA> What color is the shirt?\n<QA Response> blue\n<QA> What is the style of the shirt?\n<QA Response> floral\n<QA> What color is the person's hair?\n<QA Response> blonde \n<DONE> It looks like you are wearing a blue floral shirt. It looks nice with your blonde hair!",
            },
            {
                "query": "What do you think about this shirt?",
                "response": "<QA> What color is the shirt?\n<QA Response> blue\n<QA> What is the style of the shirt?\n<QA Response> floral\n<DONE> It looks like you are wearing a blue floral shirt. It looks nice!",
            },
        ]

    def create_example_store(self):
        embeddings = OpenAIEmbeddings()
        self.example_store = MaxMarginalRelevanceExampleSelector.from_examples(
            # This is the list of examples available to select from.
            self.example_pairs,
            # This is the embedding class used to produce embeddings which are used to measure semantic similarity.
            embeddings,
            # This is the VectorStore class that is used to store the embeddings and do a similarity search over.
            FAISS,
            # This is the number of examples to produce.
            k=3,
        )
        self.mmr_prompt = FewShotPromptTemplate(
            # We provide an ExampleSelector instead of examples.
            example_selector=self.example_store,
            example_prompt=self.example_template,
            prefix="Examples:",
            suffix=" {query}",
            input_variables=["query"],
        )

    def get_examples(self, query: str):
        examples = str(self.mmr_prompt.format(query=query)).replace(query, "")
        return examples


if __name__ == "__main__":
    ditto_rag_example_store = DittoImageRAGExampleStore()
