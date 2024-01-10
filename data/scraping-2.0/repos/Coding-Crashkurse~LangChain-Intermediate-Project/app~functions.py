import os

from app.db import Session, Pizza, Order, Review
from app.prompts import QA_PROMPT
import json
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from app.store import get_vectorstore


def get_pizza_info(pizza_name: str):
    session = Session()
    pizza = session.query(Pizza).filter(Pizza.name == pizza_name).first()
    session.close()
    if pizza:
        return json.dumps(pizza.to_json())
    else:
        return "Pizza not found"


def create_order(pizza_name: str):
    session = Session()
    pizza = session.query(Pizza).filter(Pizza.name == pizza_name).first()
    if pizza:
        order = Order(pizza=pizza)
        session.add(order)
        session.commit()
        session.close()
        return "Order created"
    else:
        session.close()
        return "Pizza not found"


def create_review(review_text: str):
    session = Session()
    review = Review(review=review_text)
    session.add(review)
    session.commit()
    session.close()
    return "Review created"


def ask_vector_db(question: str):
    llm = OpenAI(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=get_vectorstore().as_retriever(),
        chain_type_kwargs={"prompt": QA_PROMPT},
    )
    result = qa.run(question)
    return result


api_functions = {
    "create_review": create_review,
    "create_order": create_order,
    "get_pizza_info": get_pizza_info,
    "ask_vector_db": ask_vector_db,
}


### Just for initialisation
def create_pizzas():
    session = Session()

    pizzas = {
        "Margherita": 7.99,
        "Pepperoni": 8.99,
        "BBQ Chicken": 9.99,
        "Hawaiian": 8.49,
        "Vegetarian": 7.99,
        "Buffalo": 9.49,
        "Supreme": 10.99,
        "Meat Lovers": 11.99,
        "Taco": 9.99,
        "Seafood": 12.99,
    }

    for name, price in pizzas.items():
        pizza = Pizza(name=name, price=price)
        session.add(pizza)

    session.commit()
    session.close()
