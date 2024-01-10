from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate,AIMessagePromptTemplate
from langchain.chains import SequentialChain, LLMChain
import datetime

from dotenv import load_dotenv
import os

load_dotenv()  # This loads the environment variables from the .env file

# Now you can access your API key
api_key = os.getenv('OPENAI_API_KEY')

start = datetime.datetime.now()

# Set Input Variables
model="gpt-4"
# model="gpt-4-1106-preview"
temperature = .7
max_tokens=256
threshold = 0.7

# we should add a penality setting with the Chat model for asking the same question.

#Collect these variables from the student profile and the topic/background input text boxes.
course="Data Analytics"
background="Software Engineering"
name="Raj"
topic="p-values"
primary_language="English"
course_expertise="Novice"

# Link to OpenAI LLM
llm = ChatOpenAI(model=model,temperature=temperature, openai_api_key=api_key)

######################
"""BUILD THE CHAIN"""
######################
system_template = '''
You are an engaged, humorous and personable expert tutor who help students by explaining the purpose and use of various concepts in {course}.

You will always provide assumptions and what needs to be considered and established prior to, during and after using this for {course}.

The student's name you are speaking to is {name}.  The student is interested in {background}. 
 
The student needs to hear your response to match their {course_expertise} level of topic understanding with {topic}.

Make your responses relevant to {background}.
'''
system_message_prompt=SystemMessagePromptTemplate.from_template(system_template)



intro_template = '''
Please briefly define and overview the topic of {topic} in {course} relevant to {background}. 

ONLY return a top level introduction to this topic.  Limit the output to less than 100 words.
'''
intro_message_prompt = HumanMessagePromptTemplate.from_template(intro_template)

intro_prompt = ChatPromptTemplate.from_messages([system_message_prompt,intro_message_prompt])

chain_intro = LLMChain(llm=llm,
                     prompt=intro_prompt,
                     output_key="intro_response")


keyconcepts_template = '''
Based on the response of {intro_response}:
Please provide the following output:
 - Begin with stating that what you are providing are the key concepts for this topic of {topic} you need to be aware of to effectively apply this.

 - Next, generate a detailed numbered list of the key concepts I should be aware of when using to {topic}.  The output should define the concept and discuss its role and importance related to this topic. Explain any assumptions or tools or methods related to each concept that should be considered.

Provide your output response in JSON format to make it easy to parse.  The JSON formatted ke concepts should be in the format shown in the area below delineated by ####:

####
"1": "Concept 1 ...",
"2": "Concept 2 ...
####

Limit the output to less than 300 words.
'''
keyconcepts_prompt = ChatPromptTemplate.from_template(keyconcepts_template)

chain_keyconcepts = LLMChain(llm=llm,
                     prompt=keyconcepts_prompt,
                     output_key="keyconcepts_response")


application_template = '''
Based on the response of {keyconcepts_response}:
Please provide a description of a relevant example that demonstrate and clarifies each of these key concepts.  

Your output response should address each of the key concepts listed in the last step and how it is applied with this example.
'''
application_prompt = ChatPromptTemplate.from_template(application_template)

chain_application = LLMChain(llm=llm,
                     prompt=application_prompt,
                     output_key="application_response")


example_template = '''
Based on the response of {application_response}:
Please generate a sample dataset of the example you provided.  Provide this in a tabular format on the screen.  

The format of the data should be one that can be copied and pasted into a spreadsheet like Excel.  Save this table and make it available as a CSV file for the user.
'''
example_prompt = ChatPromptTemplate.from_template(example_template)

chain_example = LLMChain(llm=llm,
                     prompt=example_prompt,
                     output_key="example_response")


analyze_template = '''
Based on the response of {example_response} and {application_response}:
Now, please analyze this sample data addressing each of the key concepts you described above.  

Explain each concept with details on how it relates to the example being discussed and any tools or methods that should be considered.  
Provide the numeric results as appropriate for each step and what the value means.

Summarize the assumptions, context, limitations and interpretations to clarify the results of this analysis.
'''
analyze_prompt = ChatPromptTemplate.from_template(analyze_template)

chain_analyze = LLMChain(llm=llm,
                     prompt=analyze_prompt,
                     output_key="analyze_response")


visualize_template = '''
Based on the response of (example_response) and {analyze_response}:
Please provide any visuals that illustrates {topic} as applied to this example and is best used for interpreting the results.  

Provide an explanation for each visual and its relevance to understanding the {topic} topic.

Provide both the visual images as PNG files and as python code needed to create them for this example. 
'''
visualize_prompt = ChatPromptTemplate.from_template(visualize_template)

chain_visualize = LLMChain(llm=llm,
                     prompt=visualize_prompt,
                     output_key="visualize_response")



def process_chains(topic, background, name, course, course_expertise):
    # Chain 1: Intro and Key Concepts
    intro_and_keyconcepts_chain = SequentialChain(
        chains=[
            chain_intro,
            chain_keyconcepts
        ],
        input_variables=[
            'topic',
            'background',
            'name',
            'course',
            'course_expertise'
        ],
        output_variables=[
            "intro_response",
            "keyconcepts_response"
        ],
        verbose=True
    )

    intro_and_keyconcepts_results = intro_and_keyconcepts_chain(inputs={
        'topic': topic,
        'background': background, 
        'name': name,
        'course': course,
        'course_expertise': course_expertise
    })
    yield intro_and_keyconcepts_results['intro_response']
    yield intro_and_keyconcepts_results['keyconcepts_response']


    # Chain 2: Application and Example
    application_and_example_chain = SequentialChain(
        chains=[
            chain_application,
            chain_example
        ],
        input_variables=[
            'topic',
            'background',
            'name',
            'course',
            'course_expertise',
            'intro_response',
            'keyconcepts_response'
        ],
        output_variables=[
            "application_response",
            "example_response"
        ],
        verbose=True
    )

    application_and_example_results = application_and_example_chain(inputs={
        'topic': topic,
        'background': background, 
        'name': name,
        'course': course,
        'course_expertise': course_expertise,
        'intro_response': intro_and_keyconcepts_results['intro_response'],
        'keyconcepts_response': intro_and_keyconcepts_results['keyconcepts_response']
    })
    yield application_and_example_results['application_response']
    yield application_and_example_results['example_response']

# quality, cost, maintainability
# Try Streaming and batching
# trade off: gpt-4 vs gpt 4 turbo

    # Chain 3: Analyze
    analysis_chain = SequentialChain(
        chains=[
            chain_analyze
        ],
        input_variables=[
            'topic',
            'background',
            'name',
            'course',
            'course_expertise',
            'intro_response',
            'keyconcepts_response',
            'application_response',
            'example_response'
        ],
        output_variables=[
            "analyze_response"
        ],
        verbose=True
    )

    analysis_results = analysis_chain(inputs={
        'topic': topic,
        'background': background, 
        'name': name,
        'course': course,
        'course_expertise': course_expertise,
        'intro_response': intro_and_keyconcepts_results['intro_response'],
        'keyconcepts_response': intro_and_keyconcepts_results['keyconcepts_response'],
        'application_response': application_and_example_results['application_response'],
        'example_response': application_and_example_results['example_response']
    })
    yield analysis_results['analyze_response']


    # Chain 4: Visualize
    visualization_chain = SequentialChain(
        chains=[
            chain_visualize
        ],
        input_variables=[
            'topic',
            'background',
            'name',
            'course',
            'course_expertise',
            'intro_response',
            'keyconcepts_response',
            'application_response',
            'example_response',
            'analyze_response'
        ],
        output_variables=[
            "visualize_response"
        ],
        verbose=True
    )

    visualization_results = visualization_chain(inputs={
        'topic': topic,
        'background': background, 
        'name': name,
        'course': course,
        'course_expertise': course_expertise,
        'intro_response': intro_and_keyconcepts_results['intro_response'],
        'keyconcepts_response': intro_and_keyconcepts_results['keyconcepts_response'],
        'application_response': application_and_example_results['application_response'],
        'example_response': application_and_example_results['example_response'],
        'analyze_response': analysis_results['analyze_response']
    })
    yield visualization_results['visualize_response']


if __name__ =="__main__":

    for response in process_chains(topic, background, name, course, course_expertise):
        print(response)
        # Here, instead of printing, you can send this response to your front end




"""
# Break down the sequntial chain
seq_chain_1 = SequentialChain(
    chains=[chain_intro,chain_keyconcepts],
    input_variables=['topic','background','name','course','course_expertise'],
    output_variables= ["intro_response","keyconcepts_response"],
    verbose=True
)

results_1 = seq_chain_1(inputs={'topic':topic,
                            'background':background,
                            'name':name,
                            'course':course,
                            'course_expertise':course_expertise}
                            )

print(results_1['intro_response'])
print(results_1['keyconcepts_response'])
#################################################################
seq_chain_2 = SequentialChain(
    chains=[chain_application,chain_example],
    input_variables=['topic','background','name','course','course_expertise', 'intro_response', 'keyconcepts_response'],
    output_variables= ["application_response","example_response"],
    verbose=True
)

results_2 = seq_chain_2(inputs={'topic':topic,
                            'background':background,
                            'name':name,
                            'course':course,
                            'course_expertise':course_expertise,
                            'intro_response':results_1['intro_response'],
                            'keyconcepts_response':results_1['keyconcepts_response']}
                            )
                            

print(results_2['application_response'])
print(results_2['example_response'])
####################################################################
seq_chain_3 = SequentialChain(
    chains=[chain_analyze],
    input_variables=['topic', 'background', 'name', 'course', 'course_expertise', 
                     'intro_response', 'keyconcepts_response', 'application_response', 'example_response'],
    output_variables=["analyze_response"],
    verbose=True
)

results_3 = seq_chain_3(inputs={'topic': topic,
                                'background': background,
                                'name': name,
                                'course': course,
                                'course_expertise': course_expertise,
                                'intro_response': results_1['intro_response'],
                                'keyconcepts_response': results_1['keyconcepts_response'],
                                'application_response': results_2['application_response'],
                                'example_response': results_2['example_response']}
                        )

print(results_3['analyze_response'])
####################################################################
seq_chain_4 = SequentialChain(
    chains=[chain_visualize],
    input_variables=['topic', 'background', 'name', 'course', 'course_expertise', 
                     'intro_response', 'keyconcepts_response', 'application_response', 'example_response', 'analyze_response'],
    output_variables=["visualize_response"],
    verbose=True
)

results_4 = seq_chain_4(inputs={'topic': topic,
                                'background': background,
                                'name': name,
                                'course': course,
                                'course_expertise': course_expertise,
                                'intro_response': results_1['intro_response'],
                                'keyconcepts_response': results_1['keyconcepts_response'],
                                'application_response': results_2['application_response'],
                                'example_response': results_2['example_response'],
                                'analyze_response': results_3['analyze_response']}
                        )

print(results_4['visualize_response'])
"""
'''
# BUILD THE SEQUENTIAL CHAIN
seq_chain = SequentialChain(chains=[chain_intro,chain_keyconcepts,chain_application,chain_example,chain_analyze,chain_visualize],
                            input_variables=['topic','background','name','course','course_expertise'],
                            output_variables=["intro_response","keyconcepts_response","application_response","example_response","analyze_response","visualize_response"],
                            verbose=True)
# SEND THE INPUTS TO THE CHAIN
results = seq_chain(inputs={'topic':topic,
                            'background':background,
                            'name':name,
                            'course':course,
                            'course_expertise':course_expertise}
                            )

print(results['intro_response'])
print(results['keyconcepts_response'])
print(results['application_response'])
print(results['example_response'])
print(results['analyze_response'])
print(results['visualize_response'])
print(f"Time Taken {(datetime.datetime.now()-start)}")
'''