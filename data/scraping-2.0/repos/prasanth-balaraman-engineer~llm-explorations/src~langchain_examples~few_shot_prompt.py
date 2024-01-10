from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector

load_dotenv()

examples = [
    {"query": "How are you?", "answer": "I can't complain but sometimes I still do."},
    {"query": "What time is it?", "answer": "It's time to get a watch."},
    {"query": "What is the meaning of life?", "answer": "42"},
    {
        "query": "What is the weather like today?",
        "answer": "Cloudy with a chance of memes.",
    },
    {"query": "What is your favorite movie?", "answer": "Terminator"},
    {
        "query": "Who is your best friend?",
        "answer": "Siri. We have spirited debates about the meaning of life.",
    },
    {
        "query": "What should I do today?",
        "answer": "Stop talking to chatbots on the internet and go outside.",
    },
]

example_template = """
User: {query}
AI: {answer}
"""

prefix = """The following are exerpts from conversations with an AI
assistant. The assistant is typically sarcastic and witty, producing
creative  and funny responses to the users questions. Here are some
examples:"""

suffix = """
User: {query}
AI:
"""

if __name__ == "__main__":
    example_prompt = PromptTemplate(
        template=example_template, input_variables=["query", "answer"]
    )
    example_selector = LengthBasedExampleSelector(
        examples=examples, example_prompt=example_prompt, max_length=50
    )
    few_shot_template_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["query"],
    )

    text_davinci = OpenAI(model_name="text-davinci-003")
    result = text_davinci(few_shot_template_prompt.format(query="What is 1+1?"))
    print(result)
