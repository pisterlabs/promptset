from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate

prompt_template = """You are an excellent executive assistent. \
    You are given an excerpt of a machine generated transcript of a stand-up meeting of the formal methods group at MPI-Berlin. \
    The script sometimes contains words that do not fit the context or misspelled names \
    and may need to be replaced accordingly to make sense.
    {text}
    Based on the excerpt please identify all relevant talking points and agreed upon action items. \
    Do NOT make up any points. For context: Team members regulalry attending the meeting include: \
    Aaron Peikert, Timo von Oertzen, Hannes Diemerling, Leonie Hagitte, Maximilian Ernst, Valentin Kriegmair, Leo Kosanke, Ulman Lindenberger, Moritz Ketzer and Nicklas Hafiz.
    Tone: formal
    Format: 
    - Concise and detailed meeting summary
    - Participants: <participants>
    - Discussed: <Discussed-items>
    - Follow-up actions: <a-list-of-follow-up-actions-with-owner-names>
    Tasks:
    - Highlight who is speaking, action items, dates and agreements
    - Step by step list all points of each speaker or group of speakers
    - Work step by step.
    CONCISE SUMMARY IN ENGLISH:"""

init_proompt = PromptTemplate(template=prompt_template, input_variables=["text"])

refine_template = (
        "Your are an excellent executive assistent \n"
        "Your job is to produce a final summary\n"
        "We have provided an existing meeting summary up to a certain point: {existing_answer}\n"
        "We have the opportunity to refine the existing summary"
        "(only if needed) with some more context below.\n"
        "------------\n"
        "{text}\n"
        "------------\n"
        f"Given the new context, refine the original summary: following the format"
        "Participants: <Participants>"
        "Discussed: <Discussed-items>"
        "Follow-up actions: <a-list-of-follow-up-actions-with-owner-names>"
        "Highlight who is speaking, action items, dates and agreements"
        "List points for each section by a speaker or the group in detail."
        "Work step by step."
        "If the context isn't useful, return the original summary, do NOT make anything up. Highlight agreements and follow-up actions and owners."
)

refine_prompt = PromptTemplate(
        input_variables=["existing_answer", "text"],
        template=refine_template,
)

bullet_prompt = ChatPromptTemplate.from_template(
    "You are an excellent executive assistant.\
    Your task is to create a detailed bulleted summary of the following meeting notes lossing as little meaning as possible.\
    Please format your response as markdown code. Highlight datees and agreed upon actions. \
    Summary: {summary}\
    Format: \
    '## Meeting Summary \
    ### Participants\
    ### Discussed\
    - **<Participant 1>:** <participants Message>\
        - point 1\
        - point 2\
        - ...\
    - **<Participant 2>:** <participants Message>\
        - point 1\
        - point 2\
        - ...\
    ### Action Items\
    - <a-list-of-follow-up-actions-with-owner-names>\
    ### Side Comments <if any were made>'"
)