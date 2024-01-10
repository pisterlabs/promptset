"""
Note: Checkout the method header for process_chains() to see how to use the script.
"""

from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
from time import time

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = "ls__ab0da2b88e6743e48e12dc5a8c9c5b22"
os.environ["LANGCHAIN_PROJECT"] = "Explain_the_Concept_LCEL"

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


# Input parameters - KEEP THE VARIABLES NAMES SAME AND READ IN THE VALUES FROM THE FRONT END
course = "Data Analytics"
background = "Software Engineering"
name = "Raj"
topic = "p-values"
primary_language = "English"
course_expertise = "Novice"

# Define pass-through runnables for input variables
course_runnable = RunnablePassthrough()
background_runnable = RunnablePassthrough()
topic_runnable = RunnablePassthrough()
name_runnable = RunnablePassthrough()
primary_language_runnable = RunnablePassthrough()
course_expertise_runnable = RunnablePassthrough()

intro_response_runnable = RunnablePassthrough()
keyconcepts_response_runnable = RunnablePassthrough()
application_response_runnable = RunnablePassthrough()
example_response_runnable = RunnablePassthrough()
analyze_response_runnable = RunnablePassthrough()

# Model Parameters
model = "gpt-4-1106-preview"#"gpt-4"  # "gpt-4" # "gpt-4-1106-preview"
temperature = 0.7
max_tokens = 256
threshold = 0.7
# Set Up Model
llm = ChatOpenAI(model=model, temperature=temperature, openai_api_key=api_key)
output_parser = StrOutputParser()


########## Prompt Templates
system_template = """
You are an engaged, humorous and personable expert tutor who help students by explaining the purpose and use of various concepts in {course}.

You will always provide assumptions and what needs to be considered and established prior to, during and after using this for {course}.

The student's name you are speaking to is {name}.  The student is interested in {background}. 
 
The student needs to hear your response to match their {course_expertise} level of topic understanding with {topic}.

Make your responses relevant to {background}.
"""
system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)


intro_template = """
Please briefly define and overview the topic of {topic} in {course} relevant to {background}. 

ONLY return a top level introduction to this topic.  Limit the output to less than 100 words.
"""
intro_message_prompt = HumanMessagePromptTemplate.from_template(intro_template)

intro_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, intro_message_prompt]
)


keyconcepts_template = """
Based on the response of {intro_response}:
Please provide the following output:
 - Begin with stating that what you are providing are the key concepts for this topic of {topic} you need to be aware of to effectively apply this.

 - Next, generate a detailed numbered list of the key concepts I should be aware of when using to {topic}.  The output should define the concept and discuss its role and importance related to this topic. Explain any assumptions or tools or methods related to each concept that should be considered.

Provide your output response in JSON format to make it easy to parse.  The JSON formatted key concepts should be in the format shown in the area below delineated by ####:

####
"1": "Concept 1 ...",
"2": "Concept 2 ...
####

Limit the output to less than 300 words.
"""
keyconcepts_prompt = ChatPromptTemplate.from_template(keyconcepts_template)


application_template = """
Based on the response of {keyconcepts_response}:
Please provide a relevant example that demonstrate and clarifies each of these key concepts. Keep in mind that the student has a background in {background}.

Your output response should address each of the key concepts listed in the last step and how it is applied with this example.
"""
application_prompt = ChatPromptTemplate.from_template(application_template)


example_template = """
Based on the response of {application_response}:
Please generate a sample dataset of the example you provided.  Provide this in a tabular format on the screen.  

The format of the data should be one that can be copied and pasted into a spreadsheet like Excel.  In the end, return the same data in csv format as well so that the user can copy and paste it into a CSV file.
"""
example_prompt = ChatPromptTemplate.from_template(example_template)


analyze_template = """
Based on the response of "{example_response}" and "{application_response}":
Now, please analyze this sample data addressing each of the key concepts you described in {keyconcepts_response}.  

Explain each concept with details on how it relates to the example being discussed and any tools or methods that should be considered.  
Provide the numeric results as appropriate for each step and what the value means.

Summarize the assumptions, context, limitations and interpretations to clarify the results of this analysis.
"""
analyze_prompt = ChatPromptTemplate.from_template(analyze_template)


visualize_template = """
Based on the response of {example_response} and {analyze_response}:
Please provide any visuals that illustrates {topic} as applied to this example scenario, and the analysis provided above, such that the student can learn how to interpret a real life scenario like this. 

Provide an explanation for each visual and its relevance to understanding the {topic} topic.

Provide python code needed to create the visual plots for this example. 
"""
visualize_prompt = ChatPromptTemplate.from_template(visualize_template)
########## End of Prompt Templates


# Build the LCEL Chain
##############################################
def process_chains():
    """
    Processes a series of chains, written in LCEL format, to help a student learn a particular topic
    given the following variables are set: course, background, name, topic, primary_language, course_expertise.

    Each chain in the sequence performs a specific function, starting with an introduction,
    then identifying key concepts, applying those concepts, providing examples, analyzing the example,
    and finally visualizing the results.


    Yields:
    - Each chain's response in sequence, representing the progress through the chain.

    Example Usage from front end:
    ```python
    from explain_the_concepts import process_chains

    for response in process_chains():
        print(response)
    ```

    Note: This function assumes that all necessary components and runnables (like `topic_runnable`,
    `intro_prompt`, `llm`, etc.) are already defined and properly set up.
    """
    start_time = time()  # to measure overall time taken

    ################## CHAIN 1 ############################
    start = time()
    intro_chain = (
        {
            "topic": topic_runnable,
            "background": background_runnable,
            "name": name_runnable,
            "course": course_runnable,
            "course_expertise": course_expertise_runnable,
        }
        | intro_prompt
        | ChatOpenAI(temperature=temperature, openai_api_key=api_key)
        | output_parser
    )
    intro_response = intro_chain.invoke(
        {
            "topic": topic,
            "background": background,
            "name": name,
            "course": course,
            "course_expertise": course_expertise,
        }
    )
    print(f"{'-'*40}\nIntro Response:\n{'-'*40}")
    print(intro_response)
    yield (intro_response)
    end = time()
    print(f"{'-'*20}\nTime Taken for Intro Chain: {end-start:.2f} s\n{'-'*20}\n")

    ################## CHAIN 2 ############################
    start = time()
    keyconcepts_chain = (
        {"intro_response": intro_response_runnable, "topic": topic_runnable}
        | keyconcepts_prompt
        | llm#ChatOpenAI(model = "gpt-4", temperature=temperature, openai_api_key=api_key) 
        | output_parser
    )
    keyconcepts_response = keyconcepts_chain.invoke(
        {
            "intro_response": intro_response,
            "topic": topic,
        }
    )
    yield (keyconcepts_response)
    print(f"{'-'*40}\nKeyconcepts Response:\n{'-'*40}")
    print(keyconcepts_response)
    end = time()
    print(f"{'-'*20}\nTime Taken for Keyconcepts Chain: {end-start:.2f} s\n{'-'*20}\n")

    ################## CHAIN 3 ############################
    start = time()
    application_chain = (
        {
            "keyconcepts_response": keyconcepts_response_runnable,
            "background": background_runnable,
        }
        | application_prompt
        | llm#ChatOpenAI(model = "gpt-4", temperature=temperature, openai_api_key=api_key) 
        | output_parser
    )
    application_response = application_chain.invoke(
        {"keyconcepts_response": keyconcepts_response, "background": background}
    )
    yield (application_response)
    print(f"{'-'*40}\nApplication Response:\n{'-'*40}")
    print(application_response)
    end = time()
    print(f"{'-'*20}\nTime Taken for Applications Chain: {end-start:.2f} s")

    ################## CHAIN 4 ############################
    start = time()
    example_chain = (
        {"application_response": application_response_runnable}
        | example_prompt
        | ChatOpenAI(temperature=temperature, openai_api_key=api_key)
        | output_parser
    )
    example_response = example_chain.invoke(
        {"application_response": application_response}
    )
    yield (example_response)
    print(f"{'-'*40}\nExample Response:\n{'-'*40}")
    print(example_response)
    end = time()
    print(f"{'-'*20}\nTime Taken for Example Chain: {end-start:.2f} s\n{'-'*20}\n")

    ################## CHAIN 5 ############################
    start = time()
    analyze_chain = (
        {
            "example_response": example_response_runnable,
            "application_response": application_response_runnable,
            "keyconcepts_response": keyconcepts_response_runnable,
        }
        | analyze_prompt
        | llm
        | output_parser
    )
    analyze_response = analyze_chain.invoke(
        {
            "example_response": example_response,
            "application_response": application_response,
            "keyconcepts_response": keyconcepts_response,
        }
    )
    yield (analyze_response)
    print(f"{'-'*40}\nAnalyze Response:\n{'-'*40}")
    print(analyze_response)
    end = time()
    print(f"{'-'*20}\nTime Taken for Analyze Chain: {end-start:.2f} s\n{'-'*20}\n")

    ################## CHAIN 6 ############################
    start = time()
    visualize_chain = (
        {
            "example_response": example_response_runnable,
            "analyze_response": analyze_response_runnable,
            "topic": topic_runnable,
        }
        | visualize_prompt
        | ChatOpenAI(model = "gpt-4", temperature=temperature, openai_api_key=api_key) #llm
        | output_parser
    )
    visualize_response = visualize_chain.invoke(
        {
            "example_response": example_response,
            "analyze_response": analyze_response,
            "topic": topic,
        }
    )
    yield (visualize_response)
    print(f"{'-'*40}\nVisualize Response:\n{'-'*40}")
    print(visualize_response)
    end = time()
    print(f"{'-'*20}\nTime Taken for Visualize Chain: {end-start:.2f} s\n{'-'*20}\n")

    end_time = time()
    print(f"{'*'*20}\nTotal Time Taken: {end_time-start_time:.2f} s")


if __name__ == "__main__":
    for response in process_chains():
        pass  # print(response)
        # Here, instead of printing, you can send this response to your front end
# else:
#     for response in process_chains():
#         print(response)
