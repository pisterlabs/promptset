import streamlit as st
import openai

# Function for generating the brainstorming
def brainstorming(user_input_interest, user_input_knowhow, user_input_timeframe, modeltype=None):
    # Setting the modeltype to the default if no modeltype is given
    if modeltype == None:
        modeltype = "gpt-3.5-turbo"

    # Setting the prompt for the brainstorming
    messages = [{"role": "user", 
                 "content": f"Rick and Morty visit the Data Science Retreat in Berlin and find a future data scientist, who struckles to find a portfolio project within his interests ({user_input_interest}) and his know-how ({user_input_knowhow}). And he has only {user_input_timeframe} until the presentation! Together, they iterate over some ideas until they have the perfect project, that is actually doable with the tools of only the base dimension. Write the dialogue with about 1000 tokens!"}]
    
    # Setting the ChatCompletion
    response = openai.ChatCompletion.create(
        model=modeltype,
        messages=messages
    )
    
    return response

def extract_idea(dialogue, modeltype=None):
    # Setting the modeltype to the default if no modeltype is given
    if modeltype == None:
        modeltype = "gpt-3.5-turbo"

    # Setting the prompt for the extraction
    # Setting the prompt for the project idea extraction
    messages = [{"role": "user", 
                "content": f"""In the following after "### ### ###", you will find a transcript of a brainstorming session between Rick, Morty and a future data scientist. They developed a project idea.
                
                Please extract the latest project idea from the dialogue and return the title and a brief description it in the form

                Project Title:
                <<<project_idea>>>

                Project Description:
                <<<project_description>>>

                ### ### ###

                {dialogue}
                """}]
    
    # Setting the ChatCompletion
    response = openai.ChatCompletion.create(
        model=modeltype,
        messages=messages
    )
    
    return response


# Function testing if the api key works
def is_api_key_valid():
    try:
        response = openai.Completion.create(
            engine="davinci",
            prompt="This is a test.",
            max_tokens=5
        )
    except:
        return False
    else:
        return True


# Function for extracting the project title and the project description from the dialogue
def extract_title_description(dialogue):
    """The dialogue is a string in the format
                Project Title:
                <<<project_idea>>>

                Project Description:
                <<<project_description>>>"""
    # Splitting the dialogue into the project title and the project description
    project_title = dialogue.split("Project Title:")[1].split("Project Description:")[0]
    project_description = dialogue.split("Project Description:")[1]

    # Removing line breaks at the beginning and the end of the project title and the project description
    project_title = project_title.strip()
    project_description = project_description.strip()
    
    return project_title, project_description

def generate_project_idea(user_input_interest: str, 
                          user_input_knowhow: str, 
                          user_input_timeframe: str, 
                          user_input_number_ideas: int,
                          modeltype_brainstorm: str ='gpt-4',
                          modeltype_extract: str ='gpt-3.5-turbo'):
    """
    Generates a project idea based on the user input
    
    Parameters
    ----------
    user_input_interest: str
        The interest of the user
    user_input_knowhow: str
        The know-how of the user
    user_input_timeframe: str
        The timeframe of the user
    user_input_number_ideas: int
        The number of ideas the user wants to generate
    modeltype_brainstorm: str
        The modeltype for the brainstorming
    modeltype_extract: str
        The modeltype for the extraction of the project idea from the dialogue
        
    Returns
    -------
    projects: dict
        A dictionary containing the project titles and the project descriptions in the form
        {project_title: project_description}
    tokens: dict
        A dictionary containing the tokens used in the form
        {modeltype: tokens}
    brainstorm_dialogues: dict
        A dictionary containing the brainstorming dialogue in the form
        {project_title: project_brainstorming}
    """

    project = {}
    tokens = {}
    brainstorm_dialogue = str()
    for i in range(user_input_number_ideas):
        # Generating the brainstorming dialogue
        response_brainstorm = brainstorming(user_input_interest, 
                                            user_input_knowhow, 
                                            user_input_timeframe, 
                                            modeltype=modeltype_brainstorm)
        
        # Extracting the project idea from the dialogue
        response_extract = extract_idea(response_brainstorm["choices"][0]["message"]["content"], 
                                        modeltype=modeltype_extract)
        
        # Extracting the project title and the project description from the dialogue
        project_title, project_description = extract_title_description(response_extract["choices"][0]["message"]["content"])
        
        # Adding the project title and the project description to the project dictionary
        project["Title"] = project_title
        project["Description"] = project_description

        # Adding the brainstorming dialogue to the brainstorm_dialogue dictionary
        brainstorm_dialogue = response_brainstorm["choices"][0]["message"]["content"]

        # Adding the tokens to the tokens dictionary
        if modeltype_brainstorm not in tokens:
            tokens[modeltype_brainstorm] = response_brainstorm["usage"]["total_tokens"]
        else:
            tokens[modeltype_brainstorm] += response_brainstorm["usage"]["total_tokens"]

        if modeltype_extract not in tokens:
            tokens[modeltype_extract] = response_extract["usage"]["total_tokens"]
        else:
            tokens[modeltype_extract] += response_extract["usage"]["total_tokens"]

        yield project, tokens, brainstorm_dialogue


# Making a nice looking output using streamlit
# Setting the title
st.title("Portfolio Project Generator")

# Setting the subtitle
st.subheader("Generate a Project Idea for your Data Science Portfolio")

# Setting the text
st.write("Please enter your interest, your know-how and the timeframe you have for the project.")

# Setting the input fields
user_input_interest = st.text_input("Interest", "engineering, sensors, microcontrollers")
user_input_knowhow = st.text_input("Know-how", "experimental physics, did private projects with microcontrollers")
user_input_timeframe = st.text_input("Timeframe", "2 weeks")
user_input_number_ideas = st.number_input("Number of Ideas", min_value=1, max_value=10, value=2)

user_api_key = st.text_input("OpenAI API Key", "")
# Warning for expances and security
st.warning("Use the API key at your own risk and expense. Tip: Create the key and delete it after usage.")

with st.expander("Advanced Options"):
    modeltype_brainstorm = st.selectbox("Modeltype for Brainstorming", ["gpt-4", "gpt-3.5-turbo"])
    modeltype_extract = st.selectbox("Modeltype for Idea Extraction", ["gpt-3.5-turbo", "gpt-4"])

# Setting the button
if user_input_number_ideas == 1:
    idea_or_ideas = "Idea"
else:
    idea_or_ideas = "Ideas"

if st.button(f"Generate Project {idea_or_ideas}"):

    # While waiting for the project idea to be generated, show a spinner
    with st.spinner(f"Generating Project {idea_or_ideas}..."):

        # try to set the api key
        openai.api_key = user_api_key
        if not is_api_key_valid():
            st.error("The API key is not valid. Please enter a valid API key.")
        else:
            # Generating the project ideas
            for project, tokens, brainstorm_dialogue in generate_project_idea(user_input_interest, 
                                                                                user_input_knowhow, 
                                                                                user_input_timeframe, 
                                                                                user_input_number_ideas,
                                                                                modeltype_brainstorm=modeltype_brainstorm,
                                                                                modeltype_extract=modeltype_extract):
            
                # Printing the Project Ideas
                st.subheader("Project Ideas")
                st.write(f"**{project['Title']}**")
                st.write(project["Description"])
                st.write("")

                with st.expander("Brainstorming Dialogue"):
                    st.write(brainstorm_dialogue)
                    st.write("")

                st.balloons()

            # Tokens used
            with st.expander("Tokens used"):
                for modeltype, token in tokens.items():
                    st.write(f"{modeltype}: {token}")