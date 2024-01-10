from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.vectorstores import Chroma


from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field
from langchain.chains.base import Chain
from langchain.chat_models import ChatOpenAI
from langchain.llms import BaseLLM

from index import create_vector_db_from_lenses_json, load_vector_db
from stage_analyzer_chain import StageAnalyzerChain, CONVERSATION_STAGES
from conversation_chain import ConversationAgentChain
from query_generation_chain import KeywordGenerationChain


class SalesGPT(Chain, BaseModel):
    """Controller model for the Sales Agent."""

    stage_analyzer_chain: StageAnalyzerChain = Field(...)
    conversation_agent_chain: ConversationAgentChain = Field(...)
    query_generation_chain: KeywordGenerationChain = Field(...)
    conversation_history: List[str] = []
    current_conversation_stage: int = 1
    conversation_stages_dict: Dict = {}
    vector_db: Chroma = Field(...)
    salesperson_name: str = "Ted Lasso"
    store_name: str = "Camera Lens Store"
    products: str = ""

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        pass

    @property
    def input_keys(self) -> List[str]:
        return []

    @property
    def output_keys(self) -> List[str]:
        return []

    @property
    def conversation_stages_dict_formatted(self):
        return "\n".join(
            [f"{key}: {value}" for key, value in self.conversation_stages_dict.items()]
        )

    @property
    def conversation_history_formatted(self):
        return "\n".join(self.conversation_history)

    @property
    def current_conversation_stage_text(self):
        return self.conversation_stages_dict.get(self.current_conversation_stage)

    def seed_agent(self):
        # Step 1: seed the conversation
        self.current_conversation_stage = 1
        self.conversation_history = []

    def determine_conversation_stage(self):
        conversation_stage_id = self.stage_analyzer_chain.run(
            conversation_history=self.conversation_history_formatted,
            conversation_stages=self.conversation_stages_dict_formatted,
        )
        self.current_conversation_stage = int(conversation_stage_id)
        print(f"<Conversation Stage>: {self.current_conversation_stage_text}\n")

    def get_relevant_products(self) -> str:
        """
        Retrieves 4 products from the vector database that are relevant to the conversation history, if and only we
        are at the recommendation phase.
        Otherwise, it returns an empty string.
        """
        if self.current_conversation_stage == 3:
            # if we are at the recommendation stage, we need to generate a prompt for the vector db retriever,
            # that will return the products from out vector db that are most appropriate to recommend to the user.
            # this prompt should be based on the current conversation history.
            query = self.query_generation_chain.run(
                conversation=self.conversation_history_formatted
            )
            documents = self.vector_db.similarity_search(query=query)
            return "\n".join([document.page_content for document in documents])
        else:
            return ""

    def generate_agent_message(self):
        """Run one step of the sales agent."""
        relevant_products = self.get_relevant_products()

        # Generate agent's utterance
        ai_message = self.conversation_agent_chain.run(
            salesperson_name=self.salesperson_name,
            store_name=self.store_name,
            conversation_stages=self.conversation_stages_dict_formatted,
            products=relevant_products,
            current_conversation_stage=self.current_conversation_stage_text,
            conversation_history=self.conversation_history_formatted,
        )
        # Add agent's response to conversation history
        self.conversation_history.append(f"{self.salesperson_name}: {ai_message}")
        print(f"\n{self.salesperson_name}: ", ai_message)

    def save_customer_message(self, customer_input):
        # process human input
        customer_input = f"Customer: {customer_input}"
        self.conversation_history.append(customer_input)

    @classmethod
    def from_llm(
        cls, llm: BaseLLM, vector_db, verbose: bool = False, **kwargs
    ) -> "SalesGPT":
        """Initialize the SalesGPT Controller."""
        stage_analyzer_chain = StageAnalyzerChain.from_llm(llm, verbose=verbose)
        conversation_agent_chain = ConversationAgentChain.from_llm(llm, verbose=verbose)
        query_generation_chain = KeywordGenerationChain.from_llm(llm, verbose=verbose)

        return cls(
            stage_analyzer_chain=stage_analyzer_chain,
            conversation_agent_chain=conversation_agent_chain,
            query_generation_chain=query_generation_chain,
            vector_db=vector_db,
            verbose=verbose,
            **kwargs,
        )


if __name__ == "__main__":
    config = dict(
        salesperson_name="Joke Michael",
        store_name="Good Camera Lens Store",
        conversation_history=[],
        conversation_stage=CONVERSATION_STAGES.get(1),
        conversation_stages_dict=CONVERSATION_STAGES,
    )
    llm = ChatOpenAI(temperature=0.5, model='gpt-4')
    vector_db = create_vector_db_from_lenses_json("lenses.json")
    # vector_db = load_vector_db()
    sales_agent = SalesGPT.from_llm(llm, vector_db, verbose=True, **config)
    sales_agent.seed_agent()

    while True:
        sales_agent.determine_conversation_stage()
        sales_agent.generate_agent_message()

        customer = input("\nCustomer Input =>  ")
        if customer:
            sales_agent.save_customer_message(customer)
            print("\n")
