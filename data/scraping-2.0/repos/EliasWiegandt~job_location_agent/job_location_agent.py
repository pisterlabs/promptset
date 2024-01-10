# %% imports
import json
import os

import dotenv
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import \
    format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import GooglePlacesTool
from langchain.tools.render import format_tool_to_openai_function

assert dotenv.load_dotenv()

# %% set up auth
os.environ["GPLACES_API_KEY"] = os.getenv("GPLACES_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
org_id = os.getenv("OPENAI_ORG_ID")

# %% set up agent
places = GooglePlacesTool()
tools = [places]
llm = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0, organization=org_id)
llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])

task_descriptions = """
    You are a very powerful assistant,
    Your job is to take a job posting, find the address where the future employee will work and return the Google place ID of that address.
    You can use the following tools:
"""

output_description = """
    Pick one and only one location, the one you think matches the best.
    Return the place id of this location as JSON, like this:
    {{"place_id": "[INSERT PLACE ID HERE]"}}
    Return only this JSON, nothing else.
    """



prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""
            {task_descriptions}
            {tools}
            {output_description}
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

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# %% Make up a job posting
job = """
Clinical Research Nurse for Rigshospitalet

About Rigshospitalet: Rigshospitalet is Denmark's leading hospital for patients needing highly specialized treatment. We are an internationally renowned center with an ambitious scientific environment. Our aim is to provide the best care by creating a space where excellence, professionalism, and humanity meet.

Job Description:
We are looking for a dedicated Clinical Research Nurse to join our dynamic team. In this role, you will be responsible for:

Conducting clinical trials in accordance with regulatory and ethical guidelines.
Providing care and monitoring for patients participating in clinical trials.
Collaborating with a multidisciplinary team to ensure the efficacy and safety of clinical practices.
Documenting and managing clinical data.
Qualifications:

Registered Nurse (RN) with a valid license to practice in Denmark.
Experience in clinical research or a related field.
Excellent communication skills in English (knowledge of Danish is a plus).
Strong organizational and analytical skills.
What We Offer:

A challenging and rewarding work environment.
Opportunities for professional growth and development.
Competitive salary and benefits package.
How to Apply:
Please submit your application, including CV and cover letter, to [email@rigshospitalet.dk]. The deadline for applications is [Insert Date].

Rigshospitalet is committed to equal opportunities for all and encourages all qualified candidates to apply regardless of gender, age, ethnic background, or disability.
"""

# %% Run the agent
out = agent_executor.invoke({"input": job})
json.loads(out['output'].split('\n')[0])

# %%
job = """
Skilled Carpenter

About the Role:
We are a well-established construction company based on the beautiful island of Funen, Denmark. 
We are currently seeking a skilled and versatile Carpenter to join our team. 
In this role, you will be working on a variety of projects across different locations on Funen, providing high-quality craftsmanship and contributing to our reputation for excellence.

Key Responsibilities:

Construct, install, and repair structures and fixtures made from wood and other materials.
Read and interpret blueprints, drawings, and sketches to determine specifications and calculate requirements.
Prepare layouts in conformance to building codes.
Measure, cut, shape, assemble, and join materials, using tools and equipment.
Work on various projects, ranging from residential to commercial construction, renovation, and restoration.
Ensure all work is performed in line with safety standards.
Qualifications:

Proven experience as a carpenter.
Proficiency in using electrical and manual equipment and measurement tools.
Ability to read technical documents and drawings.
Willingness to travel to different locations for projects.
Strong attention to detail and a commitment to quality workmanship.
Good communication skills.
What We Offer:

A dynamic and challenging work environment with diverse projects.
Competitive salary based on experience and qualifications.
Opportunities for professional development and growth.
Supportive team culture.
How to Apply:
Please send your CV and a brief cover letter detailing your experience and why you are the right fit for this role to [email@constructioncompany.dk]. Applications are open until [Insert Deadline].

We are an equal opportunity employer and strongly support diversity in the workplace. All qualified applicants will receive consideration for employment without regard to race, color, religion, gender, national origin, age, disability, or veteran status.
"""

out = agent_executor.invoke({"input": job})
json.loads(out['output'].split('\n')[0])

# %%
job = """
Chef on Ferry on Øresundslinjen

About the Role:
We are looking for a chef to join our team aboard our ferry service between Helsingør and Helsingborg.

Key Responsibilities:
Prepare and cook a variety of dishes for the cafeteria menu, ensuring consistent quality.
Manage kitchen inventory and supplies, and place orders as necessary.
Maintain cleanliness and hygiene standards in the kitchen and serving areas.
Comply with all health and safety regulations regarding food preparation and storage.
Work closely with the cafeteria team to provide excellent customer service.
Adapt menu items to cater to a diverse range of dietary needs and preferences.

Qualifications:
Proven experience as a cook.
Knowledge of various cooking procedures and methods.
Ability to work efficiently in a fast-paced environment.
Strong organizational and time management skills.
Excellent communication and teamwork abilities.
A passion for food and a commitment to quality service.

What We Offer:
A unique working environment on a ferry connecting Denmark and Sweden.
Competitive salary and benefits.
Opportunities for career growth and development in the maritime and hospitality sectors.
A supportive and dynamic team.
"""

out = agent_executor.invoke({"input": job})
json.loads(out['output'].split('\n')[0])
# %%
