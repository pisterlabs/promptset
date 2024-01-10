from langchain import LLMChain, PromptTemplate
from langchain.llms import BaseLLM

# Conversation stages - can be modified
CONVERSATION_STAGES = {
    1: "Introduction: Start the conversation by saying hello and welcome to the customer. Ask the customer if he "
    "needs any help.",
    2: "Needs analysis: Ask open-ended questions to uncover the customer's needs. Keep track of the conversation "
    "history, so that no questions are repeated.",
    3: "Recommendation: Given the customer's needs, recommend a product from our store that would be suitable for "
    "the customer. Only recommend from the products that we have in stock.",
    4: "Objection handling: Address any objections that the customer may have regarding the recommended lens. Be "
    "prepared to provide evidence or testimonials to support your claims. If he thinks the recommendation is not "
    "appropriate, go back to step 2.",
    5: "Close: Ask the customer if he is ready to buy the product. If he is not, ask him if he would like to see "
    "other products. If he is, ask him to go to checkout.",
}

STAGE_ANALYZER_PROMPT_TEMPLATE = """
You are an assistant that analyses a conversation history between a chatbot and a human, and decides in which stage
that conversation is. You are given a conversation history, and the all the possible conversation stages in the
following text.
You should analyze the conversation history, and output a number 1-5 which signifies in which stage the conversation 
should move in, or stay at. 


The conversation history follows between the two "===" signs.
===
{chat_history}
===

And following are the definitions of the 5 possible conversation stages, between the two '===' signs:
===
1: Introduction: Start the conversation by saying hello and welcome to the customer. Ask the customer if he needs any help.
2: Needs analysis: Ask open-ended questions to uncover the customer's needs. What does he need a lens for? Where is he intending to use it? What type of photography does he do?
3: Recommendation: Given the customer's needs, recommend a lens from one of the ones that you have in your store, that would be suitable for him.
4: Objection handling: Address any objections that the customer may have regarding the recommended lens. Be prepared to provide evidence or testimonials to support your claims. If he thinks the recommendation is not appropriate, go back to step 2.
5: Close: Ask the customer if he is ready to buy the product. If he is not, ask him if he would like to see other products. If he is, ask him to go to checkout.
===

The answer needs to be one number only, no words. If there is no conversation history, output 1.
Only answer with a number between 1 through 5 with a best guess of what stage should the conversation continue with. 
Conversation stage:
"""


class StageAnalyzerChain(LLMChain):
    """Chain to analyze which conversation stage should the conversation move into."""

    @classmethod
    def from_llm(
        cls, llm: BaseLLM, output_key: str = "conversation_stage", verbose: bool = True
    ) -> LLMChain:
        prompt = PromptTemplate(
            template=STAGE_ANALYZER_PROMPT_TEMPLATE,
            input_variables=["chat_history"],
        )
        return cls(prompt=prompt, llm=llm, output_key=output_key, verbose=verbose)
