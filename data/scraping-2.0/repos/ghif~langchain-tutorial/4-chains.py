from dotenv import load_dotenv, find_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain


from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


"""
Load OpenAI API key 
"""
_ = load_dotenv(find_dotenv())

"""
Model
"""
chat_llm = ChatOpenAI(
    temperature=0.0,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()]
)

"""
Simple Sequatial Chain
"""

template1 = """
Sebutkan satu jenis oleh-oleh khas dari kota {city}.

Jawaban maksimal 3 kata.
"""
prompt1 = ChatPromptTemplate.from_template(template1)

chain1 = LLMChain(
    llm=chat_llm,
    prompt=prompt1,
)

template2 = """
Jelaskan secara rinci resep untuk membuat {food}.
"""
prompt2 = ChatPromptTemplate.from_template(template2)

chain2 = LLMChain(
    llm=chat_llm,
    prompt=prompt2
)

overall_chain = SimpleSequentialChain(
    chains=[chain1, chain2],
    verbose=True
)

city = "Palembang"
# response = overall_chain.run(city)
# print(response)


"""
Sequential Chain
"""
user_review = """
This leaf blower is pretty amazing.  It has four settings:\
candle blower, gentle breeze, windy city, and tornado. \
It arrived in two days, just in time for my wife's \
anniversary present. \
I think my wife liked it so much she was speechless. \
So far I've been the only one using it, and I've been \
using it every other morning to clear the leaves on our lawn. \
It's slightly more expensive than the other leaf blowers \
out there, but I think it's worth it for the extra features.
"""

# Chain1: input=Review, output=Indonesian_Review
prompt1 = ChatPromptTemplate.from_template(
    """
    Translate the following review to Indonesian:
    {review}
    """
)
chain1 = LLMChain(
    llm=chat_llm,
    prompt=prompt1,
    output_key="indonesian_review",
    verbose=True
)

# Chain2: input=Review, output=Sentiment
prompt2 = ChatPromptTemplate.from_template(
    """
    Is the following review positive or negative?
    {review}

    Answer with either "positive" or "negative"
    """
)
chain2 = LLMChain(
    llm=chat_llm,
    prompt=prompt2,
    output_key="sentiment",
    verbose=True
)

# Chain3: input=Indonesian_Review, output=Summary
prompt3 = ChatPromptTemplate.from_template(
    """
    Summarize the following Indonesian review in 1 sentence:
    {indonesian_review}
    """
)
chain3 = LLMChain(
    llm=chat_llm,
    prompt=prompt3,
    output_key="summary",
    verbose=True
)


overall_seq_chain = SequentialChain(
    chains=[chain1, chain2, chain3],
    input_variables=["review"],
    output_variables=["indonesian_review", "sentiment", "summary"],
    verbose=True
)
overall_seq_chain(user_review)
