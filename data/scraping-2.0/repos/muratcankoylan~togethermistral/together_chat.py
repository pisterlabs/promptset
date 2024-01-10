from langchain.llms import Together
import os

llm = Together(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=1,
    max_tokens=512,
    top_k=1,
    together_api_key = os.getenv("TOGETHER_API_KEY"))
input_ = """You are the most controversial, techno optimist and futurist thinker who writes essays like Paul Graham. \
Create an essay intro for the given question:

Explore the controversy surrounding startups that use AI to monetize user data. Is it ethical for startups to profit from user data, and what rights should users have over their personal information?"""
print(llm(input_))