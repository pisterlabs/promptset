# Description: This script uses OpenAI GPT-3 to fix a fuzz test
from langfuzz.utils import run_initial_atheris_fuzzer
import openai
import time

from sqlalchemy import create_engine, Column, Integer, String, Boolean, Text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base

def fix_code(self, code: str, output: str) -> str:
    # Use OpenAI GPT to generate a fixed version of the code
    # Modify the code to fix the problem
    # Return the updated code
    prompt=f"Fix the following code:\n{code}\nOutput: {output} \n return only valid formatted python code."

    response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages=[
                {"role": "system", "content": prompt}],
            max_tokens=1550,
            temperature=0.6,
        )
    return response["choices"][0]["message"]["content"]

def fix_fuzz_test(self, function_code, output) -> Tuple[str, str, bool]:
    sucessful_run = 'Done 2 runs'
    n = 0
    while n < 10:
        n += 1
        updated_code = self.fix_code(function_code, output)
        new_output = run_atheris_fuzzer(updated_code)
        if sucessful_run in new_output:
            print('fixed')
            return updated_code, new_output, True
        else:
            output = new_output
            time.sleep(1)
    return updated_code, output, False

Base = declarative_base()

class GeneratedFile(Base):
    __tablename__ = 'generated_files'
    id = Column(Integer, primary_key=True)
    library_name = Column(String)
    file_name = Column(String)
    function_name = Column(String)
    contents = Column(Text)
    runs = Column(Boolean)
    fuzz_test = Column(Boolean)
    type = Column(String)
    coverage = Column(Integer)
    cycles = Column(Integer)
    run_output = Column(Text)
    tokens = Column(Integer)
    crash = Column(Boolean)
    exception= Column(Boolean)



def query_generated_files():
    engine = create_engine('sqlite:///langfuzz.db')
    Base.metadata.bind = engine
    DBSession = sessionmaker(bind=engine)
    session = DBSession()

    # Query for GeneratedFiles with runs=False
    results = session.query(GeneratedFile).filter_by(runs=False).all()

    # Close the session
    session.close()

    return results