import argparse
import pickle

from langchain import OpenAI
from langchain.chains import VectorDBQAWithSourcesChain

import os
from dotenv import load_dotenv
load_dotenv()

def main():

    argPrser = argparse.ArgumentParser(description='yourwebsite.com Q&A')
    argPrser.add_argument('question', type=str, help='Your question for yourwebsite.com')
    args = argPrser.parse_args()

    # Load the Faiss store
    with open("vector_store.pkl", "rb") as file:
        store = pickle.load(file)

    # Create a VectorDBQAWithSourcesChain
    chain = VectorDBQAWithSourcesChain.from_llm(
        llm=OpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), temperature=0, verbose=True), vectorstore=store, verbose=True)

    # Get the result from the chain
    result = chain({"question": args.question})

    # Format the answer in a chatbot-like style
    chatbot_response = f"🤖 Q: {args.question}\n📣 A: {result['answer']}"

    # Print the chatbot-like response and sources
    print(chatbot_response)
    print(f"\nSources: {result['sources']}")

if __name__ == "__main__":
    main()
