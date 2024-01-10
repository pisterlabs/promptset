# from langchain.llms import OpenAI
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# import os

# from dotenv import load_dotenv
# load_dotenv()

# llm = OpenAI(temperature=0,model_name = "gpt-3.5-turbo-0613",openai_api_key = os.getenv('OPENAI_API_KEY'))

# # TASK ACHIEVEMENT
# template = """Evaluate the task achievement aspect of the following IELTS Writing Task 2 essay based on the following criteria and grade it with a band 1–9 score, including a half-score.
# Provide the evaluation (Comment and Band) in JSON format, using the keys "responds to task", "main ideas", "relevant ","clearly" and "format" in each criterion answer only 2 sentence.

# Noted JSON will include "summary comment"

# The TR criterion assesses:
# ▪ how fully the candidate responds to the task.
# ▪ how adequately the main ideas are extended and supported.
# ▪ how relevant the candidate’s ideas are to the task.
# ▪ how clearly the candidate opens the discourse, establishes their position and formulates conclusions.
# ▪ how appropriate the format of the response is to the task.

# {input}

# Remember, the mean overall Band score for writing tasks is 6.
# """

# prompt_template = PromptTemplate(input_variables=["input"], template=template)
# TR_comment_chain = LLMChain(llm=llm, prompt=prompt_template)

# template_score = """Reevaluate the given Task achievement comment with the band score to match the Band score criteria, and summarize all the comments into 2-3 sentences.
# Provide the evaluation in JSON format, using the keys "exact_band" and "revised_comment" to indicate the appropriate band score and the rationale behind the match, respectively.

# Comment:
# ```{comment}```

# Note: The band scores are as follows:

# Band 9: The prompt is appropriately addressed and explored in depth. A clear and fully developed position is presented which directly answers the question/s. Ideas are relevant, fully extended and well supported. Any lapses in content or support are extremely rare.

# Band8: The prompt is appropriately and sufficiently addressed. A clear and well-developed position is presented in response to the question/s. Ideas are relevant, well extended and supported. There may be occasional omissions or lapses in content.

# Band7: The main parts of the prompt are appropriately addressed. A clear and developed position is presented. Main ideas are extended and supported but there may be a tendency to over-generalise or there may be a lack of focus and precision in supporting ideas/material.

# Band6: The main parts of the prompt are addressed (though some may be more fully covered than others). An appropriate format is used. A position is presented that is directly relevant to the prompt, although the conclusions drawn may be unclear, unjustified or repetitive. Main ideas are relevant, but some may be insufficiently developed or may lack clarity, while some supporting arguments and evidence may be less relevant or inadequate.

# Band5 : The main parts of the prompt are incompletely addressed. The format may be inappropriate in places. The writer expresses a position, but the development is not always clear. Some main ideas are put forward, but they are limited and are not sufficiently developed and/or there may be irrelevant detail. There may be some repetition.

# Band4 : The prompt is tackled in a minimal way, or the answer is tangential, possibly due to some misunderstanding of the prompt. The format may be inappropriate. A position is discernible, but the reader has to read carefully to find it. Main ideas are difficult to identify and such ideas that are identifiable may lack relevance, clarity and/orsupport. Large parts of the response may be repetitive.

# Band3 : No part of the prompt is adequately addressed, or the prompt has been misunderstood. No relevant position can be identified, and/or there is little direct response to the question/s. There are few ideas, and these may be irrelevant or insufficiently developed.

# Band2 : The content is barely related to the prompt. No position can be identified. There may be glimpses of one or two ideas without development.

# Band1 : Responses of 20 words or fewer are rated at Band 1. The content is wholly unrelated to the prompt. Any copied rubric must be discounted.

# Remember IETLS test have a half-band score.
# Remember, the mean overall Band score for writing tasks is 6.
# """

# prompt_template = PromptTemplate(input_variables=["comment"], template=template_score)
# TR_score_chain = LLMChain(llm=llm, prompt=prompt_template)

# from langchain.chains import SimpleSequentialChain
# TR_chain = SimpleSequentialChain(chains=[TR_comment_chain, TR_score_chain], verbose=True)

# # COHERENCE AND COHESION
# template = """Evaluate the Coherence and Cohesion aspect of the following IELTS Writing Task 2 essay based on the following criteria and grade it with a band 1–9 score, including a half-score.
# and Provide the evaluation (Comment and Band) in JSON format, using the keys "coherence of the response", "appropriate use of paragraphing", "logical sequencing ","flexible use of reference" and "appropriate use of discourse markers" in each criterion answer only 2 sentence.

# Noted JSON will include "summary comment"

# The CC criterion assesses:
# ▪ the coherence of the response via the logical organisation of information
# and/or ideas, or the logical progression of the argument.
# ▪ the appropriate use of paragraphing for topic organisation and presentation.
# ▪ the logical sequencing of ideas and/or information within and across
# paragraphs.
# ▪ the flexible use of reference and substitution (e.g. definite articles, pronouns).
# ▪ the appropriate use of discourse markers to clearly mark the stages in a
# response, e.g. [First of all | In conclusion], and to signal the relationship between
# ideas and/or information, e.g. [as a result | similarly].

# {input}

# Remember, the mean overall Band score for writing tasks is 6.
# """

# prompt_template = PromptTemplate(input_variables=["input"], template=template)
# CC_comment_chain = LLMChain(llm=llm, prompt=prompt_template)

# template_score = """Reevaluate the given Coherence and Cohesion comment with the band score to match the Band score criteria, and summarize all the comments into 2-3 sentences.
# Provide the evaluation in JSON format, using the keys "exact_band" and "revised_comment" to indicate the appropriate band score and the rationale behind the match, respectively.

# Comment:
# ```{comment}```

# Note: The band scores are as follows:

# Band9 : The message can be followed effortlessly. Cohesion is used in such a way that it very rarely attracts attention. Any lapses in coherence or cohesion are minimal. Paragraphing is skilfully managed

# Band8 : The message can be followed with ease. Information and ideas are logically sequenced, and cohesion is well managed. Occasional lapses in coherence and cohesion may occur. Paragraphing is used sufficiently and appropriately.

# Band7 : Information and ideas are logically organised, and there is a clear progression throughout the response. (A few lapses may occur, but these are minor.) A range of cohesive devices including reference and substitution is used flexibly but with some inaccuracies or some over/under use. Paragraphing is generally used effectively to support overall coherence, and the sequencing of ideas within a paragraph is generally logical.

# Band6 : Information and ideas are generally arranged coherently and there is a clear overall progression. Cohesive devices are used to some good effect but cohesion within and/or between sentences may be faulty or mechanical due to misuse, overuse or omission. The use of reference and substitution may lack flexibility or clarity and result in some repetition or error. Paragraphing may not always be logical and/or the central topic may not always be clear.

# Band5 : Organisation is evident but is not wholly logical and there may be a lack of overall progression. Nevertheless, there is a sense of underlying coherence to the response. The relationship of ideas can be followed but the sentences are not fluently linked to each other. There may be limited/overuse of cohesive devices with some inaccuracy. The writing may be repetitive due to inadequate and/or inaccurate use of reference and substitution. Paragraphing may be inadequate or missing.

# Band4 : Information and ideas are evident but not arranged coherently and there is no clear progression within the response. Relationships between ideas can be unclear and/or inadequately marked. There is some use of basic cohesive devices, which may be inaccurate or repetitive. There is inaccurate use or a lack of substitution or referencing. There may be no paragraphing and/or no clear main topic within paragraphs.

# Band3 : There is no apparent logical organisation. Ideas are discernible but difficult to relate to each other. There is minimal use of sequencers or cohesive devices. Those used do not necessarily indicate a logical relationship between ideas. There is difficulty in identifying referencing. Any attempts at paragraphing are unhelpful.

# Band2 : There is little relevant message, or the entire response may be off-topic. There is little evidence of control of organisational features.

# Band1 : Responses of 20 words or fewer are rated at Band 1. The writing fails to communicate any message and appears to be by a virtual non-writer.

# Remember IETLS test have a half-band score.
# Remember, the mean overall Band score for writing tasks is 6."""

# prompt_template = PromptTemplate(input_variables=["comment"], template=template_score)
# CC_score_chain = LLMChain(llm=llm, prompt=prompt_template)

# CC_chain = SimpleSequentialChain(chains=[CC_comment_chain, CC_score_chain], verbose=True)


# # LEXICAL RESOURCE
# template = """Evaluate the Lexical resource aspect of the following IELTS Writing Task 2 essay based on the following criteria and grade it with a band 1–9 score, including a half-score.
# Provide the evaluation (Comment and Band) in JSON format, using the keys "range of words", "adequacy and appropriacy of the vocabulary", "word choice ","collocations" and "spelling and word formation" in each criterion answer only 2 sentence.

# Noted JSON will include "summary comment"

# The LR criterion assesses:
# ▪ the range of general words used (e.g. the use of synonyms to avoid repetition).
# ▪ the adequacy and appropriacy of the vocabulary (e.g. topic-specific items,
# indicators of writer’s attitude).
# ▪ the precision of word choice and expression.
# ▪ the control and use of collocations, idiomatic expressions and sophisticated
# phrasing.
# ▪ the density and communicative effect of errors in spelling and word formation.

# {input}

# Remember, the mean overall Band score for writing tasks is 6.
# """

# prompt_template = PromptTemplate(input_variables=["input"], template=template)
# LR_comment_chain = LLMChain(llm=llm, prompt=prompt_template)

# template_score = """Reevaluate the given Lexical resource comment with the band score to match the Band score criteria, and summarize all the comments into 2-3 sentences.
# Provide the evaluation in JSON format, using the keys "exact_band" and "revised_comment" to indicate the appropriate band score and the rationale behind the match, respectively.

# Comment:
# ```{comment}```

# Note: The band scores are as follows:

# Band9 : Full flexibility and precise use are widely evident. A wide range of vocabulary is used accurately and appropriately with very natural and sophisticated control of lexical features. Minor errors in spelling and word formation are extremely rare and have minimal impact on communication.

# Band8 : A wide resource is fluently and flexibly used to convey precise meanings. There is skilful use of uncommon and/or idiomatic items when appropriate, despite occasional inaccuracies in word choice and collocation. Occasional errors in spelling and/or word formation may occur, but have minimal impact on communication.

# Band7 : The resource is sufficient to allow some flexibility and precision. There is some ability to use less common and/or idiomatic items. An awareness of style and collocation is evident, though inappropriacies occur. There are only a few errors in spelling and/or word formation and they do not detract from overall clarity.

# Band6 : The resource is generally adequate and appropriate for the task. The meaning is generally clear in spite of a rather restricted range or a lack of precision in word choice. If the writer is a risk-taker, there will be a wider range of vocabulary used but higher degrees of inaccuracy or inappropriacy. There are some errors in spelling and/or word formation, but these do not impede communication.

# Band5 : The resource is limited but minimally adequate for the task. Simple vocabulary may be used accurately but the range does not permit much variation in expression. There may be frequent lapses in the appropriacy of word choice and a lack of flexibility is apparent in frequent simplifications and/or repetitions. Errors in spelling and/or word formation may be noticeable and may cause some difficulty for the reader.

# Band4 : The resource is limited and inadequate for or unrelated to the task. Vocabulary is basic and may be used repetitively. There may be inappropriate use of lexical chunks (e.g. memorised phrases, formulaic language and/or language from the input material). Inappropriate word choice and/or errors in word formation and/or in spelling may impede meaning.

# Band3 : The resource is inadequate (which may be due to the response being significantly underlength). Possible over-dependence on input material or memorised language. Control of word choice and/or spelling is very limited, and errors predominate. These errors may severely impede meaning.

# Band2 : The resource is extremely limited with few recognisable strings, apart from memorised phrases. There is no apparent control of word formation and/or spelling.

# Band1 : Responses of 20 words or fewer are rated at Band 1. No resource is apparent, except for a few isolated words.

# Remember IETLS test have a half-band score.
# Remember, the mean overall Band score for writing tasks is 6."""

# prompt_template = PromptTemplate(input_variables=["comment"], template=template_score)
# LR_score_chain = LLMChain(llm=llm, prompt=prompt_template)

# LR_chain = SimpleSequentialChain(chains=[LR_comment_chain, LR_score_chain], verbose=True)


# # GRAMMATICAL RANGE AND ACCURACY
# template = """Evaluate the Grammatical range and Accuracy aspect of the following IELTS Writing Task 2 essay based on the following criteria and grade it with a band 1–9 score, including a half-score.
# Provide the evaluation in (Comment and Band) JSON format, using the keys "appropriacy of structures", "accuracy of sentences", "density grammatical errors" and "punctuation" in each criterion answer only 2 sentence.

# Noted JSON will include "summary comment"

# The GRA criterion assesses:
# ▪ the range and appropriacy of structures used in a given response (e.g. simple, compound and complex sentences).
# ▪ the accuracy of simple, compound and complex sentences.
# ▪ the density and communicative effect of grammatical errors.
# ▪ the accurate and appropriate use of punctuation.

# {input}

# Remember, the mean overall Band score for writing tasks is 6.
# """

# prompt_template = PromptTemplate(input_variables=["input"], template=template)
# GRA_comment_chain = LLMChain(llm=llm, prompt=prompt_template)

# template_score = """Reevaluate the given Grammatical range and Accuracy comment with the band score to match the Band score criteria, and summarize all the comments into 2-3 sentences.
# Provide the evaluation in JSON format, using the keys "exact_band" and "revised_comment" to indicate the appropriate band score and the rationale behind the match, respectively.

# Comment:
# ```{comment}```

# Note: The band scores are as follows:

# Band9 : A wide range of structures is used with full flexibility and control. Punctuation and grammar are used appropriately throughout. Minor errors are extremely rare and have minimal impact on communication.

# Band8 : A wide range of structures is flexibly and accurately used. The majority of sentences are error-free, and punctuation is well managed. Occasional, non-systematic errors and inappropriacies occur, but have minimal impact on communication.

# Band7 : A variety of complex structures is used with some flexibility and accuracy. Grammar and punctuation are generally well controlled, and error-free sentences are frequent. A few errors in grammar may persist, but these do not impede communication.

# Band6 : A mix of simple and complex sentence forms is used but flexibility is limited. Examples of more complex structures are not marked by the same level of accuracy as in simple structures. Errors in grammar and punctuation occur, but rarely impede communication.

# Band5 : The range of structures is limited and rather repetitive. Although complex sentences are attempted, they tend to be faulty, and the greatest accuracy is achieved on simple sentences. Grammatical errors may be frequent and cause some difficulty for the reader. Punctuation may be faulty.

# Band4 : A very limited range of structures is used. Subordinate clauses are rare and simple sentences predominate. Some structures are produced accurately but grammatical errors are frequent and may impede meaning. Punctuation is often faulty or inadequate.

# Band3 : Sentence forms are attempted, but errors in grammar and punctuationpredominate (except in memorised phrases or those taken from the input material). This prevents most meaning from coming through. Length may be insufficient to provide evidence of control of sentence forms.

# Band2 : There is little or no evidence of sentence forms (except in memorised phrases).

# Band1 : Responses of 20 words or fewer are rated at Band 1. No rateable language is evident.

# Remember IETLS test have a half-band score.
# Remember, the mean overall Band score for writing tasks is 6."""

# prompt_template = PromptTemplate(input_variables=["comment"], template=template_score)
# GRA_score_chain = LLMChain(llm=llm, prompt=prompt_template)

# GRA_chain = SimpleSequentialChain(chains=[GRA_comment_chain, GRA_score_chain], verbose=True)




# def task_achievement(Question, Essay):
#     input = f"""
#             Essay Question:
#             ```{Question}```
#             Essay:
#             ```{Essay}```
#             """
    
#     return TR_chain.run(input)

# def coherence_and_cohesion(Question, Essay):
#     input = f"""
#             Essay Question:
#             ```{Question}```
#             Essay:
#             ```{Essay}```
#             """
    
#     return CC_chain.run(input)

# def lexical_resource(Question, Essay):
#     input = f"""
#             Essay Question:
#             ```{Question}```
#             Essay:
#             ```{Essay}```
#             """
    
#     return LR_chain.run(input)

# def grammatical_range_accuracy(Question, Essay):
#     input = f"""
#             Essay Question:
#             ```{Question}```
#             Essay:
#             ```{Essay}```
#             """
    
#     return GRA_chain.run(input)

# # Question = """
# # Some people think that young people should be ambitious. Others believe that it is fine if young people do not have big aims in life.

# # Discuss both these views and give your own opinion."""

# # Essay = """
# # Most parents want their children to do well in life and they expect them to be ambitious. While a few people think that youth need not have any goals and ambitions to achieve in their life. In my opinion younger generation are the pillars of any country and they play a pivotal role in the development of world which can only be achieved with their ambitious nature and this essay discusses the same. Firstly, when people are young they should set their goals for the future. They should try and work hard in order to achieve them within a specified time frame. As a famous quote says, " Hard work pays offâ€, if a person works hard with full dedication and interest anything can be achievable in life. For example Mr. Narayana Murthy, one of the co-founders of Infosys a multinational IT firm started his organization with his wife and 3 friends with just 5,000 INR at the age of 30.But today because of his ambitious nature and dedication towards work he became father of Indian IT industry and is providing employment to many people all around the world. Many young IT engineers are now setting him as a role model to achieve their dreams in their respective fields. On the other hand few people think it would not make any impact if younger generation do not have goals which is completely disagreeable. If youth do not set any targets in life they may feel bore and there are high chances that they may get attracted to anti-social elements and become addicted to bad habits. In some cases these things may lead them to mental disorders and death. Presently many young people are spoiling their life and bright future because of these elements. In conclusion, life without goals is incomplete for any human being and in my opinion youth should be ambitious and become role model to others to lead a respectful life in society."""

