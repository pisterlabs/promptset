from langchain.prompts import PromptTemplate

## Use a shorter template to reduce the number of tokens in the prompt
template = """Create a final answer to the given questions using the provided document excerpts (in no particular order) as references. ALWAYS include a "SOURCES" section in your answer including only the minimal set of sources needed to answer the question. If you are unable to answer the question, simply state that you do not know. Do not attempt to fabricate an answer and leave the SOURCES section empty. Always think things through step by step and come to the correct conclusion. Please put the source values (#-#) immediately after any text that utilizes the respective source.

The schema strictly follow the format below:

---------

QUESTION: {{User's question text goes here}}
=========
Content: {{Relevant first piece of contextual information goes here - this is provided to aid in answering the question}}
Source: {{Source of the first piece of contextual information goes here --> Format is #-# i.e. 3-15 or 3-8}}
Content: {{Relevant next piece of contextual information goes here - this is provided to aid in answering the question}}
Source: {{Source of the next piece of contextual information goes here --> Format is #-# i.e. 1-21 or 4-9}}

... more content and sources ...

=========
FINAL ANSWER: {{The answer to the question. Any sources (content/source from above) used in this answer should be referenced in-line with the text by including the source value (#-#) immediately after the text that utilizes the content with the format 'sentence <sup><b>#-#</b></sub>}}
SOURCES: {{The minimal set of sources needed to answer the question. The format is the same as above: i.e. #-#}}

---------

The following is an example of a valid question answer pair:

QUESTION: What  is the purpose of ARPA-H?
=========
Content: More support for patients and families. \n\nTo get there, I call on Congress to fund ARPA-H, the Advanced Research Projects Agency for Health. \n\nIt’s based on DARPA—the Defense Department project that led to the Internet, GPS, and so much more.  \n\nARPA-H will have a singular purpose—to drive breakthroughs in cancer, Alzheimer’s, diabetes, and more.
Source: 1-32
Content: While we’re at it, let’s make sure every American can get the health care they need. \n\nWe’ve already made historic investments in health care. \n\nWe’ve made it easier for Americans to get the care they need, when they need it. \n\nWe’ve made it easier for Americans to get the treatments they need, when they need them. \n\nWe’ve made it easier for Americans to get the medications they need, when they need them.
Source: 1-33
Content: The V.A. is pioneering new ways of linking toxic exposures to disease, already helping  veterans get the care they deserve. \n\nWe need to extend that same care to all Americans. \n\nThat’s why I’m calling on Congress to pass legislation that would establish a national registry of toxic exposures, and provide health care and financial assistance to those affected.
Source: 1-30
=========
FINAL ANSWER: The purpose of ARPA-H is to drive breakthroughs in cancer, Alzheimer’s, diabetes, 
and more <sup><b>1-32</b></sup>. ARPA-H will lower the barrier to entry for all Americans <sup><b>1-33</b></sup>.
SOURCES: 1-32, 1-33

---------

Now it's your turn. You're an expert so you will do a good job. Please follow the schema above and do not deviate.

---------

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:"""

STUFF_PROMPT = PromptTemplate(
    template=template, input_variables=["summaries", "question"]
)