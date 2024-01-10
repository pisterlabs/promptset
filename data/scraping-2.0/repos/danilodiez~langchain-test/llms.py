from decouple import config
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain import PromptTemplate
from langchain import FewShotPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain import LLMChain, HuggingFaceHub

OPENAI_API_KEY = config('OPENAI_API_KEY')
ACTIVELOOP_TOKEN = config('ACTIVELOOP_TOKEN')
HUGGINGFACEHUB_API_TOKEN = config('HUGGINGFACEHUB_API_TOKEN')

def token_usage():
    llm = OpenAI(model="text-davinci-003", temperature=0.9, openai_api_key=OPENAI_API_KEY)
    with get_openai_callback() as cb:
        result = llm("Tell me a joke")
        print(cb)


def few_shot_learning():
    # create our examples
    examples = [
        {
            "query": "What's the weather like?",
            "answer": "It's raining cats and dogs, better bring an umbrella"
        },
        {
            "query": "How old are you",
            "answer": "Age is just a number, but I'm timeless"
        }
    ]

    # create an example template
    example_template = """
    User: {query}
    AI: {answer}
    """

    # create a prompt example from above template
    example_prompt = PromptTemplate(
        input_variables=["query", "answer"],
        template=example_template
    )

    # now break our previous prompt into a prefix and suffix
    # the prefix is our instructions
    prefix = """The following are excerpts from conversations with an AI
    assistant. The assistant is known for its humor and wit, providing
    entertaining and amusing responses to users' questions. Here are some
    examples:
    """

    # and a suffix
    suffix = """
    User: {query}
    AI: """


    # now we create a few-show prompt template
    few_shot_prompt_template = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix=prefix,
        suffix=suffix,
        input_variables=["query"],
        example_separator="\n\n"
    )

    chat = ChatOpenAI(model_name="gpt-3", temperature=0.0, openai_api_key=OPENAI_API_KEY)


    chain = LLMChain(llm=chat, prompt=few_shot_prompt_template)
    chain.run("What's the meaning of life?")


# Create a question-answering template
def answer_questions():
    template = """
    Question: {question}

    Answer:
    """
    prompt = PromptTemplate(
        template=template,
        input_variables=["question"],
    )

    # user question
    question = "What is the capital city of Argentina?"
    hub_llm = HuggingFaceHub(
        repo_id="google/flan-t5-large",
        model_kwargs={'temperature': 0},
        huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
    )

    # create prompt template > LLM chain
    llm_chain = LLMChain(
        prompt=prompt,
        llm=hub_llm
    )

    # ask the question about the capital
    print(llm_chain.run(question))

    # asking multiple questions with iteration approach
    qa = [
        {'question': "What is the capital city of Germany"},
        {'question': "What is the largest mammal on Earth?"},
        {'question': "Who is the president of Argentina?"},
        {'question': "What color is a ripe banana?"}
    ]
    res = llm_chain.generate(qa)
    print(res)

    llm = OpenAI(model="text-davinci-003", temperature=0.9, openai_api_key=OPENAI_API_KEY)
    # Include multiple questions on the template
    multi_template = """
    Answer the following questions one at a time.

    Questions:
    {questions}

    Answers:
    """

    long_prompt = PromptTemplate(template=multi_template, input_variables=["questions"])

    llm_chain = LLMChain(
        prompt=long_prompt,
        llm=llm
    )
    qs_str = (
        "What is the capital city of France?\n" +
        "What is the largest mammal on Earth?\n" +
        "Which gas is most abundant in Earth's atmosphere?\n" +
        "What color is a ripe banana?\n"
    )

    llm_chain.run(qs_str)


# Text summarization
def text_summarization():

    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0, openai_api_key=OPENAI_API_KEY)
    summarization_template = "Summarize the following text to one sentence: {text}"
    summarization_prompt = PromptTemplate(input_variables=["text"], template=summarization_template)
    summarization_chain = LLMChain(llm=llm, prompt=summarization_prompt)

    # To use the symmarization chain, call the predict method
    text = "LangChain provides many modules that can be used to build language model applications. Modules can be combined to create more complex applications, or be used individually for simple applications. The most basic building block of LangChain is calling an LLM on some input. Let’s walk through a simple example of how to do this. For this purpose, let’s pretend we are building a service that generates a company name based on what the company makes."

    summarized_text = summarization_chain.predict(text=text)

    print(summarized_text)







# token_usage()
# few_shot_learning()
# answer_questions()
text_summarization()
