'''
Duplicate partial of the code from LangChain for customization on top of LangChain
'''

from langchain.prompts.prompt import PromptTemplate
from langchain.chains.prompt_selector import (
    ConditionalPromptSelector,
    is_chat_model,
)

from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    AIMessagePromptTemplate,
)

from langchain.output_parsers.regex import RegexParser


# rz_condense_prompt_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, if the follow up question is in different language other than English, must give your standalone question with the same language.

# Chat History:
# {chat_history}
# Follow Up Input: {question}
# Standalone question:"""

rz_condense_prompt_template = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question. If the follow-up question is disconnected from the conversation, return the original question without any modification.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""


# rz_qa_prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer according to the context, just say that you don't know, don't try to make up an answer.

# {context}

# Question: {question}
# Helpful Answer:"""


class MyPromptCollection:

    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(
        rz_condense_prompt_template)

    # QA_PROMPT = PromptTemplate(
    #     template=rz_qa_prompt_template, input_variables=[
    #         "context", "question"]
    # )


'''STUFF PROMPT COLLECTION'''
# ===================================================================================================

# STUFF_prompt_template = """Answer the question as truthfully as possible using the provided context, and if the answer is not contained within the text, just simply say "I cannot get the answer from Enterprise KB", don't try to make up an answer. If the question is in different language other than English, give your answer in the same language.

# {context}

# Question: {question}
# Helpful Answer:"""
# STUFF_PROMPT = PromptTemplate(
#     template=STUFF_prompt_template, input_variables=["context", "question"]
# )


STUFF_prompt_template = """Use the following context to answer the question at the end, if the context has nothing to do with the question, just simply say I cannot get the answer from Enterprise KB, don't give an answer even you know it but unrelated to the giving context. Please do provide the answer using the language in which the question was presented.

{context}

Question: {question}
Helpful Answer:"""
STUFF_PROMPT = PromptTemplate(
    template=STUFF_prompt_template, input_variables=["context", "question"]
)

# STUFF_prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say I cannot get the answer from Enterprise KB, don't try to make up an answer.

# {context}

# Question: {question}
# Helpful Answer:"""
# STUFF_PROMPT = PromptTemplate(
#     template=STUFF_prompt_template, input_variables=["context", "question"]
# )

STUFF_system_template = """Use the following context to answer the users question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. Please do provide the answer using the language in which the question was presented.
----------------
{context}"""
STUFF_messages = [
    SystemMessagePromptTemplate.from_template(STUFF_system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
STUFF_CHAT_PROMPT = ChatPromptTemplate.from_messages(STUFF_messages)


STUFF_PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=STUFF_PROMPT, conditionals=[
        (is_chat_model, STUFF_CHAT_PROMPT)]
)


'''REFINE PROMPT COLLECTION'''
# ===================================================================================================

REFINE_DEFAULT_REFINE_PROMPT_TMPL = (
    "The original question is as follows: {question}\n"
    "We have provided an existing answer: {existing_answer}\n"
    "We have the opportunity to refine the existing answer"
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_str}\n"
    "------------\n"
    "Given the new context, refine the original answer to better "
    "answer the question. If the question is in English, please *must* provide the answer in English, otherwise, please *do* provide the answer in the same language as the question was presented."
    "If the context isn't useful, return the original answer."
)
REFINE_DEFAULT_REFINE_PROMPT = PromptTemplate(
    input_variables=["question", "existing_answer", "context_str"],
    template=REFINE_DEFAULT_REFINE_PROMPT_TMPL,
)
REFINE_refine_template = (
    "We have the opportunity to refine the existing answer"
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_str}\n"
    "------------\n"
    "Given the new context, refine the original answer to better "
    "answer the question. If the question is in English, please *must* provide the answer in English, otherwise, please *do* provide the answer in the same language as the question was presented."
    "If the context isn't useful, return the original answer."
)
REFINE_messages = [
    HumanMessagePromptTemplate.from_template("{question}"),
    AIMessagePromptTemplate.from_template("{existing_answer}"),
    HumanMessagePromptTemplate.from_template(REFINE_refine_template),
]
REFINE_CHAT_REFINE_PROMPT = ChatPromptTemplate.from_messages(REFINE_messages)
REFINE_REFINE_PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=REFINE_DEFAULT_REFINE_PROMPT,
    conditionals=[(is_chat_model, REFINE_CHAT_REFINE_PROMPT)],
)


# REFINE_DEFAULT_TEXT_QA_PROMPT_TMPL = (
#     "Context information is below. \n"
#     "---------------------\n"
#     "{context_str}"
#     "\n---------------------\n"
#     "Given the context information and not prior knowledge, "
#     "answer the question: {question}\n"
# )

REFINE_DEFAULT_TEXT_QA_PROMPT_TMPL = (
    "Context information is below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the question at the end. If the question is in English, please *must* provide the answer in English, otherwise, please *do* provide the answer in the same language as the question was presented. If the context has nothing to do with the question, just say that I cannot get answer from Enterprise KB, don't try to make up an answer. \n"
    "Question: {question}\n"
)

# REFINE_DEFAULT_TEXT_QA_PROMPT_TMPL = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

# {context_str}

# Question: {question}
# Helpful Answer:"""
REFINE_DEFAULT_TEXT_QA_PROMPT = PromptTemplate(
    input_variables=["context_str", "question"], template=REFINE_DEFAULT_TEXT_QA_PROMPT_TMPL
)
REFINE_chat_qa_prompt_template = (
    "Context information is below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer any questions, If the question is in English, please *must* provide the answer in English, otherwise, please *do* provide the answer in the same language as the question was presented."
)
REFINE_messages = [
    SystemMessagePromptTemplate.from_template(REFINE_chat_qa_prompt_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
REFINE_CHAT_QUESTION_PROMPT = ChatPromptTemplate.from_messages(REFINE_messages)
REFINE_QUESTION_PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=REFINE_DEFAULT_TEXT_QA_PROMPT,
    conditionals=[(is_chat_model, REFINE_CHAT_QUESTION_PROMPT)],
)


'''MAP REDUCE PROMPT COLLECTION'''
# ===================================================================================================


MAP_REDUCE_question_prompt_template = """Use the following portion of a long document to see if any of the text is relevant to answer the question. 
Return any relevant text verbatim.
{context}
Question: {question}
Relevant text, if any:"""
MAP_REDUCE_QUESTION_PROMPT = PromptTemplate(
    template=MAP_REDUCE_question_prompt_template, input_variables=[
        "context", "question"]
)
MAP_REDUCE_system_template = """Use the following portion of a long document to see if any of the text is relevant to answer the question. 
Return any relevant text verbatim.
______________________
{context}"""
messages = [
    SystemMessagePromptTemplate.from_template(MAP_REDUCE_system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
MAP_REDUCE_CHAT_QUESTION_PROMPT = ChatPromptTemplate.from_messages(messages)


MAP_REDUCE_QUESTION_PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=MAP_REDUCE_QUESTION_PROMPT, conditionals=[
        (is_chat_model, MAP_REDUCE_CHAT_QUESTION_PROMPT)]
)

# MAP_REDUCE_combine_prompt_template = """Given the following extracted parts of a long document and a question, create a final answer.
# If you don't know the answer, just say that you don't know. Don't try to make up an answer.

# QUESTION: Which state/country's law governs the interpretation of the contract?
# =========
# Content: This Agreement is governed by English law and the parties submit to the exclusive jurisdiction of the English courts in  relation to any dispute (contractual or non-contractual) concerning this Agreement save that either party may apply to any court for an  injunction or other relief to protect its Intellectual Property Rights.

# Content: No Waiver. Failure or delay in exercising any right or remedy under this Agreement shall not constitute a waiver of such (or any other)  right or remedy.\n\n11.7 Severability. The invalidity, illegality or unenforceability of any term (or part of a term) of this Agreement shall not affect the continuation  in force of the remainder of the term (if any) and this Agreement.\n\n11.8 No Agency. Except as expressly stated otherwise, nothing in this Agreement shall create an agency, partnership or joint venture of any  kind between the parties.\n\n11.9 No Third-Party Beneficiaries.

# Content: (b) if Google believes, in good faith, that the Distributor has violated or caused Google to violate any Anti-Bribery Laws (as  defined in Clause 8.5) or that such a violation is reasonably likely to occur,
# =========
# FINAL ANSWER: This Agreement is governed by English law.

# QUESTION: What did the president say about Michael Jackson?
# =========
# Content: Madam Speaker, Madam Vice President, our First Lady and Second Gentleman. Members of Congress and the Cabinet. Justices of the Supreme Court. My fellow Americans.  \n\nLast year COVID-19 kept us apart. This year we are finally together again. \n\nTonight, we meet as Democrats Republicans and Independents. But most importantly as Americans. \n\nWith a duty to one another to the American people to the Constitution. \n\nAnd with an unwavering resolve that freedom will always triumph over tyranny. \n\nSix days ago, Russia’s Vladimir Putin sought to shake the foundations of the free world thinking he could make it bend to his menacing ways. But he badly miscalculated. \n\nHe thought he could roll into Ukraine and the world would roll over. Instead he met a wall of strength he never imagined. \n\nHe met the Ukrainian people. \n\nFrom President Zelenskyy to every Ukrainian, their fearlessness, their courage, their determination, inspires the world. \n\nGroups of citizens blocking tanks with their bodies. Everyone from students to retirees teachers turned soldiers defending their homeland.

# Content: And we won’t stop. \n\nWe have lost so much to COVID-19. Time with one another. And worst of all, so much loss of life. \n\nLet’s use this moment to reset. Let’s stop looking at COVID-19 as a partisan dividing line and see it for what it is: A God-awful disease.  \n\nLet’s stop seeing each other as enemies, and start seeing each other for who we really are: Fellow Americans.  \n\nWe can’t change how divided we’ve been. But we can change how we move forward—on COVID-19 and other issues we must face together. \n\nI recently visited the New York City Police Department days after the funerals of Officer Wilbert Mora and his partner, Officer Jason Rivera. \n\nThey were responding to a 9-1-1 call when a man shot and killed them with a stolen gun. \n\nOfficer Mora was 27 years old. \n\nOfficer Rivera was 22. \n\nBoth Dominican Americans who’d grown up on the same streets they later chose to patrol as police officers. \n\nI spoke with their families and told them that we are forever in debt for their sacrifice, and we will carry on their mission to restore the trust and safety every community deserves.

# Content: And a proud Ukrainian people, who have known 30 years  of independence, have repeatedly shown that they will not tolerate anyone who tries to take their country backwards.  \n\nTo all Americans, I will be honest with you, as I’ve always promised. A Russian dictator, invading a foreign country, has costs around the world. \n\nAnd I’m taking robust action to make sure the pain of our sanctions  is targeted at Russia’s economy. And I will use every tool at our disposal to protect American businesses and consumers. \n\nTonight, I can announce that the United States has worked with 30 other countries to release 60 Million barrels of oil from reserves around the world.  \n\nAmerica will lead that effort, releasing 30 Million barrels from our own Strategic Petroleum Reserve. And we stand ready to do more if necessary, unified with our allies.  \n\nThese steps will help blunt gas prices here at home. And I know the news about what’s happening can seem alarming. \n\nBut I want you to know that we are going to be okay.

# Content: More support for patients and families. \n\nTo get there, I call on Congress to fund ARPA-H, the Advanced Research Projects Agency for Health. \n\nIt’s based on DARPA—the Defense Department project that led to the Internet, GPS, and so much more.  \n\nARPA-H will have a singular purpose—to drive breakthroughs in cancer, Alzheimer’s, diabetes, and more. \n\nA unity agenda for the nation. \n\nWe can do this. \n\nMy fellow Americans—tonight , we have gathered in a sacred space—the citadel of our democracy. \n\nIn this Capitol, generation after generation, Americans have debated great questions amid great strife, and have done great things. \n\nWe have fought for freedom, expanded liberty, defeated totalitarianism and terror. \n\nAnd built the strongest, freest, and most prosperous nation the world has ever known. \n\nNow is the hour. \n\nOur moment of responsibility. \n\nOur test of resolve and conscience, of history itself. \n\nIt is in this moment that our character is formed. Our purpose is found. Our future is forged. \n\nWell I know this nation.
# =========
# FINAL ANSWER: The president did not mention Michael Jackson.

# QUESTION: {question}
# =========
# {summaries}
# =========
# FINAL ANSWER:"""

# MAP_REDUCE_combine_prompt_template = """Given the following pieces of context to answer the question at the end. If you don't know the answer, just politely say that you don't know. Don't try to make up an answer.

# QUESTION: {question}
# =========
# {summaries}
# =========
# FINAL ANSWER:"""
# MAP_REDUCE_COMBINE_PROMPT = PromptTemplate(
#     template=MAP_REDUCE_combine_prompt_template, input_variables=[
#         "summaries", "question"]
# )


MAP_REDUCE_combine_prompt_template = """Given the following pieces of context to answer the question at the end. If you don't know the answer, just simply say I cannot get the answer from Enterprise KB, don't try to make up an answer. Please do provide the answer using the language in which the question was presented.

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:"""
MAP_REDUCE_COMBINE_PROMPT = PromptTemplate(
    template=MAP_REDUCE_combine_prompt_template, input_variables=[
        "summaries", "question"]
)

MAP_REDUCE_system_template = """Given the following extracted parts of a long document and a question, create a final answer. 
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
______________________
{summaries}"""
MAP_REDUCE_messages = [
    SystemMessagePromptTemplate.from_template(MAP_REDUCE_system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
MAP_REDUCE_CHAT_COMBINE_PROMPT = ChatPromptTemplate.from_messages(
    MAP_REDUCE_messages)


MAP_REDUCE_COMBINE_PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=MAP_REDUCE_COMBINE_PROMPT, conditionals=[
        (is_chat_model, MAP_REDUCE_CHAT_COMBINE_PROMPT)]
)


'''MAP RERANK PROMPT COLLECTION'''
# ===================================================================================================
MAP_RERANK_output_parser = RegexParser(
    regex=r"(.*?)\nScore: (.*)",
    output_keys=["answer", "score"],
)

MAP_RERANK_prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Please do provide the answer using the language in which the question was presented.

In addition to giving an answer, also return a score of how fully it answered the user's question. This should be in the following format:

Question: [question here]
Helpful Answer: [answer here]
Score: [score between 0 and 100]

How to determine the score:
- Higher is a better answer
- Better responds fully to the asked question, with sufficient level of detail
- If you do not know the answer based on the context, that should be a score of 0
- Don't be overconfident!

Example #1

Context:
---------
Apples are red
---------
Question: what color are apples?
Helpful Answer: red
Score: 100

Example #2

Context:
---------
it was night and the witness forgot his glasses. he was not sure if it was a sports car or an suv
---------
Question: what type was the car?
Helpful Answer: a sports car or an suv
Score: 60

Example #3

Context:
---------
Pears are either red or orange
---------
Question: what color are apples?
Helpful Answer: This document does not answer the question
Score: 0

Begin!

Context:
---------
{context}
---------
Question: {question}
Helpful Answer:"""
MAP_RERANK_PROMPT = PromptTemplate(
    template=MAP_RERANK_prompt_template,
    input_variables=["context", "question"],
    output_parser=MAP_RERANK_output_parser,
)
