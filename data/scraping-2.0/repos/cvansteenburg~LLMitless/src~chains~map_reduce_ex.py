import os
from functools import partial
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain.callbacks.tracers import ConsoleCallbackHandler
from langchain.chains.combine_documents import collapse_docs, split_list_of_docs
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document, StrOutputParser
from langchain.schema.prompt_template import format_document
from langchain.schema.runnable import (
    Runnable,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)

load_dotenv()

os.environ["LANGCHAIN_WANDB_TRACING"] = "true"
os.environ["WANDB_PROJECT"] = "langchain-tracing2"

app = FastAPI()

llm = ChatOpenAI()

doc_prompt = PromptTemplate.from_template("{page_content}")
partial_format_document = partial(format_document, prompt=doc_prompt)

# Runnables are stored as callable attributes within an instance of RunnableParallel.
# Those callable attributes are then executed concurrently by RunnableParallel.
# Here, a RunnableParallel is generated with two callable attributes.
# The runnable parallel takes a single argument, in this case a Document object, which
# is passed by RunnableParallel into both of it's callable attributes concurrently.
# map_chain's first Runnable is also a RunnableParallel (any map in a runnable defaults to RunnableParallel).
# So map_chain takes the Document and passes it to partial_format_document, which takes a Document as it's first argument
map_chain: Runnable[Any, str] = (
    {"content": partial_format_document}
    | PromptTemplate.from_template("Summarize the following content:\n\n{content}")
    | llm
    | StrOutputParser()
).with_config(run_name="Summarize (return doc)")

sum_and_recombine = (
    RunnableParallel({"doc": RunnablePassthrough(), "content": map_chain})
    | (lambda x: Document(page_content=x["content"], metadata=x["doc"].metadata))
).with_config({"run_name": "Summaries to Document"})

# The chain we'll repeatedly apply to collapse subsets of the documents
# into a consolidate document until the total token size of our
# documents is below some max size.
def combine_docs(docs):
    return "\n\n".join(partial_format_document(doc) for doc in docs)


collapse_chain: Runnable[Any, str]  = (
    {"context": combine_docs}
    | PromptTemplate.from_template("Collapse this content:\n\n{context}")
    | llm
    | StrOutputParser()
)


def get_num_tokens(docs):
    return llm.get_num_tokens(combine_docs(docs))


def _collapse(
    docs,
    config,
    token_max=500, # SET LOW FOR TEST CASE
    iteration_limit=3,
):
    collapse_ct = 1
    while get_num_tokens(docs) > token_max and collapse_ct < iteration_limit:

        # configure collapse_chain to include run number
        config["run_name"] = f"Collapse {collapse_ct}"
        collapse_chain_w_config = partial(collapse_chain.invoke, config=config)

        # create a list of lists of docs, each with content (excl. metadata) no longer than token_max (pops docs from queue until max, doesn't mix and match)
        split_docs = split_list_of_docs(docs, get_num_tokens, token_max)

        # execute collapse_chain on each list of docs
        docs = [collapse_docs(_docs, collapse_chain_w_config) for _docs in split_docs]

        collapse_ct += 1
    return docs

collapse = RunnableLambda(_collapse)


# The chain we'll use to combine our individual document summaries
# (or summaries over subset of documents if we had to collapse the map results)
# into a final summary.

reduce_chain: Runnable[Any, str] = (
    {"context": combine_docs}
    | PromptTemplate.from_template("Combine these summaries:\n\n{context}")
    | llm
    | StrOutputParser()
).with_config(run_name="Reduce")

# The final full chain
map_reduce = (sum_and_recombine.map() | collapse | reduce_chain).with_config(
    run_name="Map reduce"
).with_config({'callbacks': [ConsoleCallbackHandler()]})

text = """Nuclear power in space is the use of nuclear power in outer space, typically either small fission systems or radioactive decay for electricity or heat. Another use is for scientific observation, as in a Mössbauer spectrometer. The most common type is a radioisotope thermoelectric generator, which has been used on many space probes and on crewed lunar missions. Small fission reactors for Earth observation satellites, such as the TOPAZ nuclear reactor, have also been flown.[1] A radioisotope heater unit is powered by radioactive decay and can keep components from becoming too cold to function, potentially over a span of decades.[2]

The United States tested the SNAP-10A nuclear reactor in space for 43 days in 1965,[3] with the next test of a nuclear reactor power system intended for space use occurring on 13 September 2012 with the Demonstration Using Flattop Fission (DUFF) test of the Kilopower reactor.[4]

After a ground-based test of the experimental 1965 Romashka reactor, which used uranium and direct thermoelectric conversion to electricity,[5] the USSR sent about 40 nuclear-electric satellites into space, mostly powered by the BES-5 reactor. The more powerful TOPAZ-II reactor produced 10 kilowatts of electricity.[3]

Examples of concepts that use nuclear power for space propulsion systems include the nuclear electric rocket (nuclear powered ion thruster(s)), the radioisotope rocket, and radioisotope electric propulsion (REP).[6] One of the more explored concepts is the nuclear thermal rocket, which was ground tested in the NERVA program. Nuclear pulse propulsion was the subject of Project Orion.[7]

Regulation and hazard prevention[edit]
After the ban of nuclear weapons in space by the Outer Space Treaty in 1967, nuclear power has been discussed at least since 1972 as a sensitive issue by states.[8] Particularly its potential hazards to Earth's environment and thus also humans has prompted states to adopt in the U.N. General Assembly the Principles Relevant to the Use of Nuclear Power Sources in Outer Space (1992), particularly introducing safety principles for launches and to manage their traffic.[8]
Benefits

Both the Viking 1 and Viking 2 landers used RTGs for power on the surface of Mars. (Viking launch vehicle pictured)
While solar power is much more commonly used, nuclear power can offer advantages in some areas. Solar cells, although efficient, can only supply energy to spacecraft in orbits where the solar flux is sufficiently high, such as low Earth orbit and interplanetary destinations close enough to the Sun. Unlike solar cells, nuclear power systems function independently of sunlight, which is necessary for deep space exploration. Nuclear-based systems can have less mass than solar cells of equivalent power, allowing more compact spacecraft that are easier to orient and direct in space. In the case of crewed spaceflight, nuclear power concepts that can power both life support and propulsion systems may reduce both cost and flight time.[9]

Selected applications and/or technologies for space include:

Radioisotope thermoelectric generator
Radioisotope heater unit
Radioisotope piezoelectric generator
Radioisotope rocket
Nuclear thermal rocket
Nuclear pulse propulsion
Nuclear electric rocket
"""

docs = [
    Document(
        page_content=split,
        metadata={"source": "https://en.wikipedia.org/wiki/Nuclear_power_in_space"},
    )
    for split in text.split(sep="\n\n")
]

if __name__ == "__main__":

    print(map_reduce.invoke(docs, config={"max_concurrency": 3}))