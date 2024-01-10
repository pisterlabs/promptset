from langchain.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate

# Map
map_template = """The following is one of a set of excerpts from an machine-generated transcript of a hybrid team meeting likely containing some transcription errors and might lack coherence in parts. \n
Excerpts:`{text}`\n
Based on this list of excerpts, 
condense the transcript but only WITHOUT loosing ANY information allowing for information-loss-free reconstruction of the original transcript. \n
correct obvious transcription errors that do not match the context. \n
Note that longer segments attributed to one speaker might correspond to multiple speakers attending the hybrid meeting in person recorded as one speaker. \n
Only include speaker attribution if it is directly relevant to the discussed content otherwise refer to a speaker as "the team". Focus on information-loss-free content condensation. \n
Thus provide a detailed and cohesive meeting transcript by correcting any transcription errors. Work step by step. \n
Comprehensive Meeting Notes:"""
# map_prompt = PromptTemplate.from_template(map_template)
# map_chain = LLMChain(llm=llm, prompt=map_prompt)
map_prompt_template = PromptTemplate (
    input_variables=["text"],
    template=map_template
)

# Reduce
reduce_template = """You are an excellent executive assistant. The following is one of a set of excerpts from a meeting transcripts of a recent hybrid meeting: \n
Excerpt:`{text}`\n
Take these and in a stepwise manner condense the transcript text in one coherent document but WITHOUT loosing ANY meaning allowing for information-loss-free reconstruction of the original transcript. \n
Note that longer segments attributed to one speaker might correspond to multiple speakers attending the hybrid meeting in person recorded as one speaker. \n
To accomadte this ONLY include speaker attribution if it is DIRECTLY relevant to the discussed content otherwise refer to a speaker as "the team".\n
Focus on discussed topics and NOT participant attribution of talking points and action items unless explicitly mentioned by a speaker. \n
Thus step by step distill a condensed but information rich, complete and cohesive summary for optimal information retrieval. \n
Meeting Summary:"""
# reduce_prompt = PromptTemplate.from_template(reduce_template)
combine_prompt_template = PromptTemplate(
    input_variables=["text"],
    template=reduce_template
)


# Second Map Reduce Chain:
# Map2
map_template2 = """The following is a set of excerpts from an meeting summary of a hybrid team meeting. \n
Excerpt:`{text}`\n
Based on this list of excerpts, 
condense the transcript but only WITHOUT loosing ANY information allowing for information-loss-free reconstruction of the original transcript. \n
correct obvious transcription errors that do not match the context. \n
Only include speaker attribution if it is directly relevant to the discussed content otherwise refer to a speaker as "the team". Focus on information-loss-free content condensation. \n
Thus provide a detailed and cohesive meeting summary. Work step by step. \n
Comprehensive Meeting Notes:"""

map_prompt_template2 = PromptTemplate (
    input_variables=["text"],
    template=map_template2
)

# Reduce2
reduce_template2 = """You are an excellent executive assistant. The following is set of excerpts from a meeting transcripts of a recent hybrid meeting: \n
Excerpt:`{text}`\n
Take these and in a stepwise manner condense the transcript text in one coherent document but WITHOUT loosing ANY meaning allowing for information-loss-free reconstruction of the original transcript. \n
Note that longer segments attributed to one speaker might correspond to multiple speakers attending the hybrid meeting in person recorded as one speaker. \n
To accomadte this ONLY include speaker attribution if it is DIRECTLY relevant to the discussed content otherwise refer to a speaker as "the team".\n
Focus on discussed topics and NOT participant attribution of talking points and action items unless explicitly mentioned by a speaker. \n
Thus step by step distill a condensed but information rich, complete and cohesive summary for optimal information retrieval. \n
Meeting Summary:"""
combine_prompt_template2 = PromptTemplate(
    input_variables=["text"],
    template=reduce_template2
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


bullet_prompt = ChatPromptTemplate.from_template(
    """You are an excellent executive assistant.\n
    The following are meeting notes from a recent team meeting of the {team_name}:\n
    Summary: {summary}\n
    Your task is to create a well structured bulleted document of these meeting notes in Markdown format loosing as little meaning as possible.\n
    Instructions:\n
    - Focus on discussed topics and NOT participant attribution of talking points and action items UNLESS explicitly mentioned by a speaker. \n
    - Highlight dates and agreed upon action items. Work step by step, format your response in Markdown according to the following sample strucutre:\n
    - Correct potentially mispelled names of participants according to the following list of regular team members:\n
    - {team_members}\n
    Sample Format: \n
    Markdown:```## Meeting Summary \n
    ### Participants\n
    - <list of participants in the meeting you can identify>\n
    ### Discussed\n
    - **<First Discussion Point>:**\n
        - <sub-point 1>\n
        - <sub-point 2>\n
        - ...\
    - **<Second Discussion Point>:**\n
        - <sub-point 1>\n
        - <sub-point 2>\n
        - ...\n
    ...
    ### Action Items\n
    - <a list of follow up actions. Include owner names ONLY if applicable>\n
    ### Side Comments <trivial side comments if any were made, otherwise ommit>```"""
)