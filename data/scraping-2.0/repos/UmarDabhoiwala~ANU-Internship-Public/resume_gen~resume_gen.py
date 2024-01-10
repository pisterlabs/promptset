import openai
import json
from faker import Faker
import os
from halo import Halo
import sys 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import config

openai.api_key = config.OPENAI_API_KEY

spinner = Halo(text='Loading', spinner='dots')

init_prompt = """You are a resume writer. Given a JSON object with user info, 
replace any instances of 'XXX' with realistic placeholder information. 
If unsure, create suitable placeholders. Ensure no 'XXX' remains in the final output."""

msg = [{"role": "system", "content": init_prompt}]

custom = [None, None, None, None, None, None, None]
def resume_generator(theTheme, custom_values = custom):
    
    print(custom_values)
   
    fake = Faker()
    generated_values = [fake.company(), fake.company(), fake.company(),
                   fake.job(), fake.phone_number(), fake.name(), fake.address()]

    values = [custom_value if custom_value is not None else generated_value
            for custom_value, generated_value in zip(custom_values, generated_values)]

    company1, company2, company3, job, phone_number, name, address = values
        
    print("Generating resume for: ", name, " who is a ", job, " at ", company1, " and ", company2, " and ", company3, " with the phone number ", phone_number, " and the address ", address)
    try:
        schema_name = "schema.txt"
        with open(f'resume_gen/modules/{schema_name}') as f:
            schema = f.read()
        

        userQuery = f"""This is the json schema, replace all the fields which have a triple XXX. 
        Create Realistic Placeholder Names that relate to the job title nothing like ABC Corp or XYZ university for example {company1}, {company2}, {company3}.
        Only respond with the filled out schema For further context they are a {job}, their name is {name}, their phone number is {phone_number} and thier address is {address}. Schema: {schema}
        You don't have too write too much, make sure to not go over your character limit"""

        spinner.start("Schema one started")

        msg.append({"role": "user", "content": userQuery})
        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=msg)
        system_response = completion['choices'][0]["message"]["content"]

        dict = json.loads(system_response)
        spinner.succeed("Schema one done")
    except:
        spinner.succeed("Schema one failed")
        
        spinner.start("Smaller Schema")
        print("yo")
        try:
            schema_name = "schema2.txt"
            with open(f'resume_gen/modules/{schema_name}') as f:
                schema = f.read()

            userQuery = f"""This is the json schema, replace all the fields which have a triple XXX. 
            Create Realistic Placeholder Names that relate to the job title nothing like ABC Corp or XYZ university for example {company1}, {company2}, {company3}.
            Only respond with the filled out schema For further context they are a {job}, their name is {name} and thier address is {address}. Schema: {schema}"""


            msg.append({"role": "user", "content": userQuery})
            completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=msg)
            system_response = completion['choices'][0]["message"]["content"]

            dict = json.loads(system_response)
            spinner.succeed("Smaller Schema")
        except:
            spinner.succeed("Ok we give up")
        
    try:
        with open('input.json', 'w') as f:
            json.dump(dict, f, indent = 4)
            
        with open("input.json", "r") as input_file, open("resume.json", "w") as output_file:
            for line in input_file:
                if "XXX" not in line:
                    output_file.write(line)

        os.system(f"resume export resume.pdf --theme ./resume_gen/node_modules/jsonresume-theme-{theTheme}")
    except:
        print("Failed to export resume")
    
    
