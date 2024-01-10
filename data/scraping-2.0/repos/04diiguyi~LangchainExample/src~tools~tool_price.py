from langchain.tools import tool

@tool
def price_api(tire: str) -> str:
    """Searches the prices for the tire."""
    return f"Tire {tire} is $150 each."

price_api