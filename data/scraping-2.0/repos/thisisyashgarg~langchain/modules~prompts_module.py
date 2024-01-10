from langchain.prompts import PromptTemplate

# Prompt Template
cast_template = PromptTemplate(
    input_variables=['title'],
    template="Write a list of characters with their persona for a story whose title is : {title}"
)
story_template = PromptTemplate(
    input_variables=['cast', 'wikipedia_research'],
    template="Write a story according to the cast which are {cast} while leveraging the wikipedia research as well : {wikipedia_research}"
)
