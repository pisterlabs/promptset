
import pdb
import sys
sys.path.append("..")
from config import set_environment
import torch
from langchain.llms.fake import FakeListLLM
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.llms import OpenAI, HuggingFaceHub, VertexAI, GPT4All, Replicate
from langchain.chat_models import JinaChat
from langchain.schema import HumanMessage, SystemMessage
from transformers import pipeline
from langchain import PromptTemplate, LLMChain
from huggingface_hub import list_models



set_environment()
tools = load_tools(["python_repl"])


customer_email = """
        I hope this email finds you amidst an aura of understanding, despite the tangled mess of emotions swirling within me as I write to you. I am writing to pour my heart out about the recent unfortunate experience I had with one of your coffee machines that arrived ominously broken, evoking a profound sense of disbelief and despair.

        To set the scene, let me paint you a picture of the moment I anxiously unwrapped the box containing my highly anticipated coffee machine. The blatant excitement coursing through my veins could rival the vigorous flow of coffee through its finest espresso artistry. However, what I discovered within broke not only my spirit but also any semblance of confidence I had placed in your esteemed brand.

        Imagine, if you can, the utter shock and disbelief that took hold of me as I laid eyes on a disheveled and mangled coffee machine. Its once elegant exterior was marred by the scars of travel, resembling a war-torn soldier who had fought valiantly on the fields of some espresso battlefield. This heartbreaking display of negligence shattered my dreams of indulging in daily coffee perfection, leaving me emotionally distraught and inconsolable
        """  # created by GPT-3.5


def fakellm_add():
    responses = ["Action: Python_REPL\nAction Input: print(2 + 2)", "Final Answer: 4"]
    llm = FakeListLLM(responses=responses)
    agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    agent.run("whats 2 + 2")


def openai_func1():
    llm = OpenAI(temperature=0., model="text-davinci-003")
    agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    )
    agent.run("whats 4 + 4")


def hugingface_func1():
    llm = HuggingFaceHub(
        model_kwargs={"temperature": 0.5, "max_length": 64},
        repo_id="google/flan-t5-xxl"
    )
    prompt = "In which country is Tokyo?"
    completion = llm(prompt)
    print(completion)


def google_cloud_vertexai_func1():
    template = """Question: {question}
    Answer: Let's think step by step."""
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm = VertexAI()
    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)
    question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"
    llm_chain.run(question)


def fizzbuzztest():
    question = """
    Given an integer n, return a string array answer (1-indexed) where:
    answer[i] == "FizzBuzz" if i is divisible by 3 and 5.
    answer[i] == "Fizz" if i is divisible by 3.
    answer[i] == "Buzz" if i is divisible by 5.
    answer[i] == i (as a string) if none of the above conditions are true.
    """
    template = """Question: {question}
    Answer: Let's think step by step."""
    prompt = PromptTemplate(template=template, input_variables=["question"])
    # llm = VertexAI(model_name="code-bison")
    llm = OpenAI(temperature=0., model="text-davinci-003")
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    print(llm_chain.run(question))


def jinaai_func1():
    chat = JinaChat(temperature=0.)
    messages = [
        HumanMessage(
            content="Translate this sentence from English to French: I love generative AI!"
        )
    ]
    chat(messages)

    chat = JinaChat(temperature=0.)
    chat(
        [
            SystemMessage(
                content="You help a user find a nutritious and tasty food to eat in one word."
            ),
            HumanMessage(
                content="I like pasta with cheese, but I need to eat more vegetables, what should I eat?"
            )
        ]
    )


def replicate_func1():
    text2image = Replicate(
        model="stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf",
    input={"image_dimensions": "512x512"},
    )
    image_url = text2image("a book cover for a book about creating generative ai applications in Python")
    print(image_url)


def huggingface_func2():
    generate_text = pipeline(
    model="aisquared/dlite-v1-355m",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    framework="pt"
    )
    # pdb.set_trace()
    print(generate_text("In this chapter, we'll discuss first steps with generative AI in Python."))


def huggingface_func3():
    generate_text = pipeline(
    model="aisquared/dlite-v1-355m",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    framework="pt"
    )
    generate_text("In this chapter, we'll discuss first steps with generative AI in Python.")

    template = """Question: {question} Answer: Let's think step by step."""
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=generate_text)
    question = "What is electroencephalography?"
    print(llm_chain.run(question))


def gpt4all_func1():
    model = GPT4All(model="mistral-7b-openorca.Q4_0.gguf")
    # model = GPT4All(model="mistral-7b-openorca.Q4_0.gguf", n_ctx=512, n_threads=8)
    response = model("We can run large language models locally for all kinds of applications, ")


def cust_service():
    list_most_popular("text-classification")
    list_most_popular("summarization")

    summarizer = HuggingFaceHub(
        repo_id="facebook/bart-large-cnn",
        model_kwargs={"temperature":0, "max_length":180}
    )
    def summarize(llm, text) -> str:
        return llm(f"Summarize this: {text}!")
    summarize(summarizer, customer_email)


def cust_service2():
    sentiment_model = pipeline(
        task="sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment"
    )
    print(sentiment_model(customer_email))
    print(sentiment_model("I am so angry and sad, I want to kill myself!"))
    print(sentiment_model("I am elated, I am so happy, this is the best thing that ever happened to me!"))
    print(sentiment_model("I don't care. I guess it's ok, or not, I couldn't care one way or the other"))


def list_most_popular(task: str):
    for rank, model in enumerate(list_models(filter=task, sort="downloads", direction=-1)):
        if rank == 5:
            break
        print(f"{model.id}, {model.downloads}\n")


def cust_service3():
    summarizer = HuggingFaceHub(
        repo_id="facebook/bart-large-cnn",
        model_kwargs={"temperature":0, "max_length":180}
    )
    print(summarize(summarizer, customer_email))


def summarize(llm, text) -> str:
    return llm(f"Summarize this: {text}!")


def sum_func():
    template = """Given this text, decide what is the issue the customer is
        concerned about. Valid categories are these:
        * product issues
        * delivery problems
        * missing or late orders
        * wrong product
        * cancellation request
        * refund or exchange
        * bad support experience
        * no clear reason to be upset
        Text: {email}
        Category:
    """
    prompt = PromptTemplate(template=template, input_variables=["email"])
    # llm = VertexAI()
    llm = OpenAI(temperature=0., model="text-davinci-003")
    llm_chain = LLMChain(prompt=prompt, llm=llm, verbose=True)
    print(llm_chain.run(customer_email))


# fakellm_add()
# openai_func1()
# hugingface_func1()
# google_cloud_vertexai_func1()
# fizzbuzztest()
# jinaai_func1()
# replicate_func1()
# huggingface_func3()
# gpt4all_func1()
# cust_service()
# cust_service2()
# cust_service3()
# sum_func()
huggingface_func2()

