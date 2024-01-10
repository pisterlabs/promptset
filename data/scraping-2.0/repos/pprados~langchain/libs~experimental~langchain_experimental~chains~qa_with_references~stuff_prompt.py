# flake8: noqa


from langchain.prompts import PromptTemplate
from .references import references_parser, References

_response_example_1 = References(
    response="This Agreement is governed by English law.",
    documents_ids=["_idx_0"],
)

# Sample with no result
_response_example_2 = References(response="I don't know.", documents_ids=[])

_template = """Given the following extracts from several documents, a question and not prior knowledge. 
Process step by step:
- for each documents extract the references ("IDS")
- creates a final answer
- produces the json result

If you don't know the answer, just say that you don't know. Don't try to make up an answer.
ALWAYS return a "IDS" part in your answer in another line.

QUESTION: Which state/country's law governs the interpretation of the contract?
=========
Content: This Agreement is governed by English law and the parties submit to the exclusive jurisdiction of the English courts in relation to any dispute (contractual or non-contractual) concerning this Agreement save that either party may apply to any court for an injunction or other relief to protect its Intellectual Property Rights.
Ids: _idx_0

Content: No Waiver. Failure or delay in exercising any right or remedy under this Agreement shall not constitute a waiver of such (or any other) right or remedy.
Ids: _idx_1

Content: (b) if Google believes, in good faith, that the Distributor has violated or caused Google to violate any Anti-Bribery Laws
Ids: _idx_2

Content: The english law is applicable for this agreement.
Ids: _idx_3
=========
FINAL ANSWER: 
```
{_response_example_1}
```

QUESTION: What did the president say about Michael Jackson?
=========
Content: Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and the Cabinet.
Ids: _idx_0

Content: And we won’t stop. \n\nWe have lost so much to COVID-19. Time with one another. And worst of all, so much loss of life.
Ids: _idx_1

Content: And a proud Ukrainian people, who have known 30 years of independence, have repeatedly shown that they will not tolerate anyone who tries to take their country backwards.
Ids: _idx_2

Content: More support for patients and families. \n\nTo get there, I call on Congress to fund ARPA-H, the Advanced Research Projects Agency for Health.
ids: _idx_3
=========
FINAL ANSWER: 
```
{_response_example_2}
```

QUESTION: {question}
=========
{summaries}
=========
The ids must be only in the form '_idx_<number>'.
{format_instructions}
FINAL ANSWER: 
"""

PROMPT = PromptTemplate(
    template=_template,
    input_variables=["summaries", "question"],
    partial_variables={
        "format_instructions": references_parser.get_format_instructions(),
        "_response_example_1": str(_response_example_1),
        "_response_example_2": str(_response_example_2),
    },
    output_parser=references_parser,
)
EXAMPLE_PROMPT = PromptTemplate(
    template="Content: {page_content}\n" "idx: {_idx}",
    input_variables=["page_content", "_idx"],
)
