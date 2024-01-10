from operator import itemgetter

from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

# Example1
prompt1 = ChatPromptTemplate.from_template("what is the city {person} is from?")
prompt2 = ChatPromptTemplate.from_template(
    "what country is the city {city} in? respond in {language}"
)

model = ChatOpenAI()

chain1 = prompt1 | model | StrOutputParser()

chain2 = (
    {"city": chain1, "language": itemgetter("language")}
    | prompt2
    | model
    | StrOutputParser()
)

print(chain2.invoke({"person": "obama", "language": "english"}))
"""
Barack Obama, the 44th President of the United States, was born in Honolulu, Hawaii, which is located in the United States of America.
"""

# Example2
from langchain.schema.runnable import RunnablePassthrough

# Generates a prompt asking for a color based on a given attribute.
prompt1 = ChatPromptTemplate.from_template(
    "generate a {attribute} color. Return the name of the color and nothing else:"
)

# Asks for a fruit of a specified color.
prompt2 = ChatPromptTemplate.from_template(
    "what is a fruit of color: {color}. Return the name of the fruit and nothing else:"
)

# Requests the name of a country with a flag containing a certain color.
prompt3 = ChatPromptTemplate.from_template(
    "what is a country with a flag that has the color: {color}. Return the name of the country and nothing else:"
)

# Forms a prompt asking for the color of a specific fruit and the flag of a specific country
prompt4 = ChatPromptTemplate.from_template(
    "What is the color of {fruit} and the flag of {country}?"
)

# Extract model message
model_parser = model | StrOutputParser()

# Generating Color
color_generator = (
    {"attribute": RunnablePassthrough()} | prompt1 | {"color": model_parser}
)

# Takes a color and uses prompt2 to ask for a corresponding fruit
color_to_fruit = prompt2 | model_parser

# uses prompt3 to find a country with a flag containing that color
color_to_country = prompt3 | model_parser


question_generator = (
    color_generator | {"fruit": color_to_fruit, "country": color_to_country} | prompt4
)

prompt = question_generator.invoke("warm")
print(prompt)
"""
messages=[HumanMessage(content='What is the color of Coral. and the flag of Comoros?')]
"""

print(model.invoke(prompt))
"""
content='The color of a pomegranate is typically a deep red or maroon. The flag of Armenia consists of three horizontal bands of equal width - the top band is red, the middle band is blue, and the bottom band is orange.'
"""


# Example3
# Branching and Merging
"""
     Input
      / \
     /   \
 Branch1 Branch2
     \   /
      \ /
      Combine
"""
planner = (
    ChatPromptTemplate.from_template("Generate an argument about: {input}")
    | ChatOpenAI()
    | StrOutputParser()
    | {"base_response": RunnablePassthrough()}
)

arguments_for = (
    ChatPromptTemplate.from_template(
        "List the pros or positive aspects of {base_response}"
    )
    | ChatOpenAI()
    | StrOutputParser()
)
arguments_against = (
    ChatPromptTemplate.from_template(
        "List the cons or negative aspects of {base_response}"
    )
    | ChatOpenAI()
    | StrOutputParser()
)

final_responder = (
    ChatPromptTemplate.from_messages(
        [
            ("ai", "{original_response}"),
            ("human", "Pros:\n{results_1}\n\nCons:\n{results_2}"),
            ("system", "Generate a final response given the critique"),
        ]
    )
    | ChatOpenAI()
    | StrOutputParser()
)

chain = (
    planner
    | {
        "results_1": arguments_for,
        "results_2": arguments_against,
        "original_response": itemgetter("base_response"),
    }
    | final_responder
)

print(chain.invoke({"input": "scrum"}))
"""
While Scrum has its pros and cons, it is important to recognize that no project management framework is a one-size-fits-all solution. The cons mentioned should be considered in the context of the specific project and organization.

For example, while Scrum may have a lack of predictability, this can be mitigated by implementing effective estimation techniques and regularly reassessing and adjusting plans. Additionally, while Scrum relies on team communication, organizations can invest in improving communication practices and tools to address any gaps or issues.

Similarly, while Scrum may have limitations for large projects, organizations can adapt Scrum by implementing scaled agile frameworks like SAFe or LeSS to address complexities and dependencies.

Furthermore, while Scrum may prioritize working software over comprehensive documentation, it does not mean that documentation is disregarded entirely. Organizations can establish guidelines and processes to ensure that essential documentation is maintained alongside the iterative development.

Regarding role clarity, organizations can establish clear role definitions and ensure that team members understand their responsibilities and accountabilities. This can be achieved through effective communication and regular feedback.

Lastly, while Scrum relies on experienced Scrum Masters, organizations can invest in training and development programs to enhance the skills and knowledge of Scrum Masters, ensuring effective facilitation of the Scrum process.

In conclusion, while Scrum has its limitations, many of these can be addressed and mitigated through proper implementation, adaptation, and organizational support. It is important to carefully consider the specific project and organizational context to determine if Scrum is the right fit and to make necessary adjustments to maximize its benefits and overcome any potential drawbacks.
"""
