from langchain import OpenAI
from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text


class InfoExtraction:
    def __init__(self):
        self.llm = OpenAI(temperature=0)

        self.order_schema = Object(
            id="order_info",
            description="Order information and product preferences.",
            attributes=[
                Text(
                    id="order_id",
                    description="The unique identifier of the order",
                    examples=[("Order ID: 123456", "123456")],
                ),
                Text(
                    id="product_preference",
                    description="The customer's preference for a specific product",
                    examples=[("I prefer the blue version of the shirt", "blue shirt")],
                ),
            ],
            examples=[
                (
                    "My order ID is 789012. I really like the red shoes.",
                    [
                        {"order_id": "789012", "product_preference": "red shoes"},
                    ],
                )
            ],
            many=True,
        )

        self.chain = create_extraction_chain(llm=self.llm, node=self.order_schema)

    def extract_order_info(self, text):
        result = self.chain.run(text)
        return result["data"]

# Usage
# order_extractor = InfoExtraction()
#
# text = """### Input: hi
# ### Response:  Hi there, how can I help you with your return today?
# ### Input: my fav color is red
# ### Response:  That's a great color! Is there anything else I can help you with today?
# ### Input: my order id is 56
# ### Input: my another order id is 575
# """
# order_info = order_extractor.extract_order_info(text)
# print(str(order_info))
