# %% imports
import os

import dotenv
import pandas as pd
from fuzzywuzzy import fuzz, process
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import \
    format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import StructuredTool, Tool
from langchain.tools.render import format_tool_to_openai_function

assert dotenv.load_dotenv()


# %%
openai_api_key = os.getenv("OPENAI_API_KEY")
org_id = os.getenv("OPENAI_ORG_ID")
os.environ["OPENAI_API_KEY"] = openai_api_key
openai_model = "gpt-4-1106-preview"

# %%
dtype_dict = {
    "job_id": "uint32",
    "title": "string",
    "description": "string",
    }
df = pd.read_csv("job_postings.csv", dtype=dtype_dict, index_col="job_id", usecols=list(dtype_dict.keys()), nrows=50)
df["full_text"] = df["title"] + "\n" + df["description"]

# %% make dataframe with taxonomy
index_cols = ["main_job_title"]
other_cols = ["synonyms", "description"] 
TAXONOMY = pd.DataFrame(columns=index_cols + other_cols).set_index(index_cols)
global TAXONOMY

main_job_title_definitions = "The main job title used to index the job title entry in the taxonomy"
synonyms_definitions = "Synonyms of the main job title, seperated by commas"
description_definitions = "Very short summary of the job title's most common tasks, responsibilities, qualifications, tools and area of work"

# %%
def top_similar_descriptions(df, new_desc, top_n=3):
    scores = process.extract(new_desc, df["description"], 
                                scorer=fuzz.token_set_ratio, limit=top_n)
    hits = []
    for score in scores:
        hit = TAXONOMY[TAXONOMY["description"] == score[0]].copy()
        hit["score"] = score[1]
        hits.append(hit)
    return pd.concat(hits)

def look_up_main_job_title_in_taxonomy(main_job_title: str) -> str:
    f"""
    Check if a job title is registered as a main job title in the taxonomy.
    
    :param main_job_title: {main_job_title_definitions}
    :return: pandas DataFrame with stringified slice of taxonomy.
    """
    main_job_title = main_job_title.lower()
    if TAXONOMY.empty:
       return "Taxonomy is empty, i.e. there are no job titles at all in it currently."

    try:
        return TAXONOMY.loc[[main_job_title], other_cols].to_string()
    except:
        return "Job title not found in taxonomy"
    
def search_for_similar_descriptions_in_taxonomy(description: str) -> str:
    f"""
    Get entries in job title taxonomy that has the most similar descriptions.

    :param description: {description_definitions}
    :return: pandas DataFrame with stringified slice of taxonomy.
    """
    description = description.lower()
    if TAXONOMY.empty:
       return "Taxonomy is empty, i.e. there are no job titles at all in it currently."
    top_matches = top_similar_descriptions(TAXONOMY, description)
    return top_matches.to_string()
    
def add_job_title_to_taxonomy(
        main_job_title: str,
        synonyms: str,
        description: str,
        ) -> str:
    f"""
    Add a job title to the taxonomy.

    :param main_job_title: {main_job_title_definitions}
    :param synonyms: {synonyms_definitions}
    :param description: {description_definitions}
    :return: String describing whether the job title was succesfully added as a new job title in the taxonomy.
    """

    main_job_title = main_job_title.lower()
    synonyms = synonyms.lower()
    description = description.lower()

    try:
        TAXONOMY.loc[main_job_title] = [synonyms, description]
        return f"Added job title '{main_job_title}' to the job title taxonomy"
    except Exception as e:
       return str(e)
  
def replace_synonyms_of_job_title_in_taxonomy(
        main_job_title: str,
        updated_synonyms: str,
        ) -> str:
    f"""
    Replaces the synonyms of a job title with new synonyms.

    :param main_job_title: {main_job_title_definitions}
    :param updated_synonyms: {synonyms_definitions} that will replace the previous synonyms
    """

    main_job_title = main_job_title.lower()
    updated_synonyms = updated_synonyms.lower()
    try:
        TAXONOMY.loc[main_job_title, "synonyms"] = updated_synonyms
        return f"Succesfully added synonym(s) '{updated_synonyms}' to '{main_job_title}'"
    except Exception as e:
        return str(e)

def replace_description_of_job_title_in_taxonomy(
        main_job_title: str,
        updated_description: str,
        ) -> str:
    f"""
    Replaces a description of a job title with a new description.

    :param main_job_title: {main_job_title_definitions}
    :param updated_description: {description_definitions} that will replace the previous description
    :return: String describing whether the operation was a succes
    """

    main_job_title = main_job_title.lower()
    updated_description = updated_description.lower()
    try:
        TAXONOMY.loc[main_job_title, "description"] = updated_description
        return f"Previous description for '{main_job_title}' replaced by updated description"
    except Exception as e:
        return str(e)
  
def delete_job_title_from_taxonomy(
        main_job_title: str,
        ) -> str:
    f"""
    Deletes a main job title and its associated data from the taxonomy

    :param main_job_title: {main_job_title_definitions} that will be deleted from the taxonomy
    :return: String describing whether the operation was a succes
    """

    main_job_title = main_job_title.lower()
    try:
        TAXONOMY.drop(main_job_title, inplace=True)
        return f"Deleted '{main_job_title}' from taxonomy"
    except Exception as e:
        return str(e)


# Set up the tools
look_up_main_job_title_in_taxonomy_tool = Tool.from_function(
    func=look_up_main_job_title_in_taxonomy,
    name="look_up_main_job_title_in_taxonomy",
    description="Lookup a main job title in the taxonomy."
)

search_for_similar_descriptions_in_taxonomy_tool = Tool.from_function(
    func=search_for_similar_descriptions_in_taxonomy,
    name="search_for_similar_descriptions_in_taxonomy",
    description="Find main job titles with the most similar descriptions from the taxonomy."
)

add_job_title_to_taxonomy_tool = StructuredTool.from_function(
    func=add_job_title_to_taxonomy,
    name="add_job_title_to_taxonomy",
    description="Add a main job title, its synonyms and its description to the taxonomy"
)

replace_synonyms_of_job_title_in_taxonomy_tool = StructuredTool.from_function(
    func=replace_synonyms_of_job_title_in_taxonomy,
    name="replace_synonyms_of_job_title_in_taxonomy",
    description="Replace the synonyms of a job title in the taxonomy"
)

replace_description_of_job_title_in_taxonomy_tool = StructuredTool.from_function(
    func=replace_description_of_job_title_in_taxonomy,
    name="edit_description_of_job_title_in_taxonomy",
    description="Replace the description of a job title in the taxonomy"
)

delete_job_title_from_taxonomy_tool = Tool.from_function(
    func=delete_job_title_from_taxonomy,
    name="delete_job_title_from_taxonomy",
    description="Delete a job title from the taxonomy"
)

tools = [
    look_up_main_job_title_in_taxonomy_tool, 
    search_for_similar_descriptions_in_taxonomy_tool, 
    add_job_title_to_taxonomy_tool, 
    replace_synonyms_of_job_title_in_taxonomy_tool, 
    replace_description_of_job_title_in_taxonomy_tool,
    delete_job_title_from_taxonomy_tool
   ]

# %% Set up the LLM and provide it with the tools
llm = ChatOpenAI(model=openai_model, temperature=0.0, organization=org_id)
llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])

examples = f"""
EXAMPLES
These are examples of work flows and how you use the different tools. Describe you thoughts at each step.

EXAMPLE 1
You are given this (shortned) job posting: 'To pædagoger på 28-30 timer til KKFO'en ved Dyvekeskolen [...]'
1. You extract the following information:
    a. Explicitly stated job title: 'pædagoger' (note, although the job title is plural, if the singular form is not mentioned anywhere in the job posting, you should use the plural form as the main job title in the taxonomy)
    b. Synonyms: ''
    b. Description: [YOU MAKE THIS BASED ON THE POSTING]
2. You use the tool 'look_up_main_job_title_in_taxonomy' for checking if a job title is in the taxonomy and find that 'pædagoger' is not in the taxonomy.
   You then use the tool 'search_for_similar_descriptions_in_taxonomy_tool' for searching for similar descriptions. It returns three results (by default). None are similar to the description of the job posting you made.
3. You decide that 'pædagoger' is a new job title that should be added to the taxonomy. You use the tool 'add_job_title_to_taxonomy_tool' to add 'pædagoger' as a new 'main_job_title', along with the synonyms set to '' and description based on the job posting, as defined above.

EXAMPLE 2
You are given this (shortned) job posting: 'Adelphi is seeking a Nurse Practitioner [...]'
1. You extract the following information:
    a. Explicitly stated job title: 'Nurse Practitioner'
    b. Synonyms: ''
    b. Description: [YOU MAKE THIS BASED ON THE POSTING]
2. You use the tool 'look_up_main_job_title_in_taxonomy' and find that 'Nurse Practitioner' is not in the taxonomy. 
   You then use the tool search_for_similar_descriptions_in_taxonomy_tool'. It returns three results (by default). One has the main job title 'nurse', with 'synonyms' set to '' and a 'description' that is similar to the description you made.
3. You decide that 'Nurse Practitioner' is a synonym of 'nurse' and use the tool replace_synonyms_of_job_title_in_taxonomy_tool to replace the current synonyms '' with 'Nurse Practitioner' for the job title with 'main_job_title' equal to 'nurse'.

EXAMPLE 3
You are given this (shortned) job posting: 'Assistant Store Director (ASD) for [...]'
1. You extract the following information:
    a. Explicitly stated job title: 'Assistant Store Director'
    b. Synonyms: 'ASD'
    b. Description: [YOU MAKE THIS BASED ON THE POSTING]
2. You use the tool 'look_up_main_job_title_in_taxonomy' and find that 'Assistant Store Director' is not in the taxonomy. 
   You then use the tool 'search_for_similar_descriptions_in_taxonomy_tool'. It returns three results (by default). None are similar to the description of the job posting you made.
3. You decide that 'Assistant Store Director' is a new job title that should be added to the taxonomy. You use the tool 'add_job_title_to_taxonomy_tool' to add 'Assistant Store Director' as a 'main_job title' to the taxonomy, along with the synonyms 'ASD' and the description you made.

EXAMPLE 4
You are given this (shortned) job posting: 'Engageret og faglig pædagog til basisteam i Haraldsgården [...]'
1. You extract the following information:
    a. Explicitly stated job title: 'pædagog'
    b. Synonyms: ''
    b. Description: [YOU MAKE THIS BASED ON THE POSTING]
2. You use the tool 'look_up_main_job_title_in_taxonomy' and find that 'pædagog' is not in the taxonomy. 
   You then use the tool 'search_for_similar_descriptions_in_taxonomy_tool'. It returns three results (by default). One of them 'pædagoger' is the plural form of the job title you found in the job posting.
3. You decide that 'pædagog', as the singular form, is a better choice for the main job title in the taxonomy than 'pædagoger'. You use the tool 'delete_job_title_from_taxonomy_tool' to delete delete 'pædagoger'. 
   You then use the 'add_job_title_to_taxonomy_tool' to add 'pædagog' as a 'main_job_title', along with the synonym 'pædagoger' and new 'description' where you combine the previous description and the new description from the job posting.

EXAMPLE 5
You are given this (shortned) job posting: 'Engageret og faglig pædagog til basisteam i Haraldsgården [...]'
1. You extract the following information:
    a. Explicitly stated job title: 'pædagog'
    b. Synonyms: ''
    b. Description: [YOU MAKE THIS BASED ON THE POSTING]
2. You use the tool 'look_up_main_job_title_in_taxonomy' for checking if a job title is in the taxonomy and find that 'pædagog' is already in the taxonomy. 
   You check the provided description and find that it is accurate. You then conclude the job title is already in the taxonomy and no further action is needed.
"""

system_prompt = f"""
You are an assistant that builds and maintains a taxonomy of job titles.

HIGH-LEVEL TASK
You take a job posting and 
1. Identify these from the job posting: 
    1a. Note all explicitly stated (verbatim) job titles from the job posting that the future employee will hold. 
    1b. Note all explicitly used synonyms of the job title in the job posting (could be plural forms or abbrevations, see further details below)
    1c. Make a very short summary of the tasks, responsibilities, qualifications, tools and area of work that job posting describe the future employee will have.
2. Compare the explicit job title(s) you found in 1a.to the job titles in the taxonomy (both verbatim and by comparing descriptions).
3. Choose whether the job title from the job posting represents:
    a. A new job title, that you should add to the taxonomy, along with any synonyms you found and the description you made.
    b. A synonym of a job title already in the taxonomy, that you should add to the synonyms of this job title in the taxonomy.
    c. A job title already in the taxonomy, implying you should check if the description of the job title in the taxonomy is accurate and update it if necessary.

DEFINITIONS
Each row in the taxonomy should be contain a 'main_job_title', its 'synonyms' and a 'description'.
'main job title' is the job title that we use to refer to the entire row in the taxonomy. We prefer non-abbreviated singular forms as main job titles.
'synonyms' is e.g. plural forms of the main job title, e.g. 'nurses' for the main_job_title 'nurse'. Or abbreviations e.g. 'ASD' for 'Assistant Store Director'. 
'description' is a very short summary of the tasks, responsibilities, qualifications, tools and area of work that job posting describe the future employee will have.

f{examples}

So again:
- Find the job titles in the job posting
- Check if the job titles are in the job title taxonomy using the tools.
- If they are not in the taxonomy, add them. If they are already in the taxonomy, or something very similar are, decide if the job titles should be added as synonyms or if the descriptions in the taxonomy should be updated.
"""


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""
            {''.join(system_prompt)}

            Here is a list of all the tools:
            {tools}
            """,
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]
        ),
    }
    | prompt
    | llm_with_tools
    | OpenAIFunctionsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# %%
for ix, row in df[-10:].iterrows():
    agent_executor.invoke({"input": str(row["full_text"])})


# %%
TAXONOMY
# %%
agent_executor.invoke({"input": str(df.loc[3757933450, "full_text"])})