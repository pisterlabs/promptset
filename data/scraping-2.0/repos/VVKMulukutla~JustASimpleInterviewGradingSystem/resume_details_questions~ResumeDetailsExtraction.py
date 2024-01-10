from PyPDF2 import PdfReader

from langchain.chat_models import ChatOpenAI
from kor import create_extraction_chain, Object, Text 
from langchain.prompts.prompt import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
import os
from dotenv import load_dotenv

#Add your own OPENAI_API_KEY to the Environment
load_dotenv()
def gen_text(pdf_file):
  #Creates Readable Text Content from a PDF Resume File.
    with open(pdf_file, "rb") as f:
        reader = PdfReader(f)
        num_pages = len(reader.pages)
        text = ""
        for page in reader.pages:
            text += page.extract_text() 
    constraints=context_extracter(text)
    return constraints
    
def context_extracter(text):
# Works with ChatGPT-3.5, Takes the Reader Resume Text File and return a JSON with Summary of Resume in it.      
      llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        max_tokens=1900,
        frequency_penalty=0,
        presence_penalty=0,
        top_p=1.0,
      )
      schema = Object(
      id="interviewer",
      description=(
        "interviewer is examining resume text and should produce set of attributes which represents that person by his resume"
        
      ),
      attributes=[
          Text(
              id="summary_or_objective",
              description="A brief overview of the candidate's professional background, skills, and career goals",
              examples=[],
              many=True,
          ),
          Text(
              id="work_experience",
              description="Details of previous employment positions, including job titles, company names, employment dates, and a description of responsibilities and achievements for each role ",
              examples=[],
              many=True,
          ),
          Text(
              id="education",
              description="Information about the candidate's educational qualifications, including degrees, certificates, and the names of institutions attended",
              examples=[],
              many=True,
          ),
          Text(
               id="skills",
               description="A section highlighting the candidate's relevant skills, such as technical skills, languages spoken, software proficiency, or specific tools used",
               examples=[],
               many=True,
          ),
          Text(
               id="achievements_or_awards",
               description="Any notable achievements, awards, or recognition received by the candidate during their education or career.",
               examples=[],
               many=True,
          ),
          Text(
               id="certifications_or_licenses",
               description="Information about any professional certifications or licenses held by the candidate that are relevant to the desired position",
               examples=[],
               many=True,
          ),
          Text(
               id="projects",
               description="Details of significant projects the candidate has worked on, including a brief description, their role, and any notable outcomes",
               examples=[],
               many=True,
          ),
          Text(
               id="publications_or_presentations",
               description=" If applicable, a list of publications or presentations the candidate has authored or delivered, including the titles, dates, and locations",
               examples=[],
               many=True,
          ),
      ],
      many=True,
      )
      # chain = LLMChain(llm=llm1, prompt=PROMPT)
      chain = create_extraction_chain(llm, schema, encoder_or_encoder_class='json')
      return chain.predict_and_parse(text=text)['data']
