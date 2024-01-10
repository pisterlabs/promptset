import re

import json
import os
import time

from openai import OpenAI
from rdf_converter import build_init_struct
# Replace 'YOUR_API_KEY' (as an ENV variable) with your actual GPT-3 API key
from pathlib import Path
class GptCodeConverter():

    MODEL_CHOICE_1 = "gpt-3.5-turbo-1106"
    MODEL_CHOICE_2 = "code-davinci-002",
    MODEL_CHOICE_3 = "gpt-3.5-turbo",
    # max_tokens=500,  # Adjust as needed
    # temperature=0.7  # Adjust the temperature for creativity

    MAX_TOKENS = 10000  # Maximum number of tokens that can be used with the OPENAI model (model dependant)

    def __init__(self, language="Java", model=MODEL_CHOICE_1):
        self.client = OpenAI(
                                # defaults to os.environ.get("OPENAI_API_KEY")
                                # api_key=api_key,
                            )
        self.model_name = model
        self.language = language
        self.results = ''
        self.system_instructions = """Create an example rdf model of the given CIM type using only the rdf, rdfs and cim schemas using the Common Information Model (CIM) prepared by the Technical Committee 57 of the IEC as a reference"""
    def create_rdf(self,  instructions):
        """
        Convert the given code snippet using GPT-3.
        """
        # Call the GPT-3 API to generate the converted code
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": self.system_instructions
                    },
                    {
                        "role": "user",
                        "content": instructions
                    }

                ],
                model=self.model_name,

            )

            # Extract and return the generated code from the response

            results = chat_completion.choices[0].message.content
        except Exception as e:
            print(e)
            results = ''
        self.results = results


if __name__ == "__main__":
    directory_path = f"{os.path.expanduser('~')}/Documents/Git/GitHub/GOSS-GridAPPS-D-PYTHON/gov_pnnl_goss/cimhub/CIM/"
    current_time = int(time.time())
    cim_types = "CIMtypes.txt"
    converter = GptCodeConverter("RDF")
    rdf_failcount = 0
    rdf_fail_files = []
    json_failcount = 0
    json_fail_files = []
    Path(directory_path).mkdir(parents=True, exist_ok=True)
    with open(directory_path + cim_types, 'r') as f:
        lines = f.readlines()
    for line in lines:
        cim_type = line.strip()
        instructions = f'Create a complex example rdf model of a {cim_type} CIM object without using xml. Make sure all rdf triples have a cim prefix.'
        print(f"Building an example rdf file for {cim_type}")
        converter.create_rdf(instructions)
        results = converter.results
        # clean up the results here
        resultant_lines = results.split('\n')
        clean_lines = []
        enclosure = False
        for r in resultant_lines:
            if enclosure and r.find("```") == 0:
                enclosure = False
                break
            if enclosure:
                # regexp to remove all these
                # line = line.replace("^^xsd:boolean", "").replace("^^xsd:float","").replace("^^xsd:int", "").\
                #             replace("^^xsd:complex", "").replace("^^xsd:integer", "").replace("^^xsd:double", "").\
                #             replace("^^xsd:string", "").replace("^^xsd:dateTime", "")  # .replace("rdf:type", "a"))

                # new_lines.append(line.replace("^^rdf:boolean", "").replace("^^rdf:float","").
                #                  replace("^^rdf:int", "").replace("^^rdf:complex", "").replace("^^rdf:integer", "").
                #                  replace("^^rdf:double", "").replace("^^rdf:string", ""))  # .replace("rdf:type", "a"))

                r2 = re.sub(r"""(\^\^[a-zA-Z0-9]*)\:([a-zA-Z0-0]*)""", "", r)
                if r2.find("@en")>0:
                    r3 = r2.replace("@en", "")
                else:
                    r3 = r2
                clean_lines.append(r3)
            if not enclosure and r.find("```") == 0:
                enclosure = True
        clean_results = '\n'.join(clean_lines)

        rdf_directory_path = f"{directory_path}rdf/"
        Path(rdf_directory_path).mkdir(parents=True, exist_ok=True)
        output_filename = f"{rdf_directory_path}{cim_type}{current_time}.rdf"
        try:
            with open(output_filename, 'w') as f2:
                f2.write(clean_results)
        except UnicodeEncodeError as e:
            rdf_failcount += 1
            print(e)
        struct_dict = {}
        json_text = "{}"
        try:
            json_directory_path = f"{directory_path}json/"
            Path(json_directory_path).mkdir(parents=True, exist_ok=True)
            output_filename = f"{json_directory_path}{cim_type}{current_time}.json"
            struct_dict = build_init_struct(cim_type, clean_results)
            json_text = json.dumps(struct_dict, indent=2)
        except Exception as e:
            print(f">>>>>>>>>> Structure build/ json.dumps failed {cim_type} error: {e}")
            json_failcount += 1
            json_fail_files.append(cim_type)
        with open(output_filename, 'w') as f2:
            f2.write(json_text)
        pjson = f"@startjson\n{json_text}\n@endjson\n"
        # Use this file name to output a non timestamped version of the CIM model.
        output_filename = f"{directory_path}puml/{cim_type}.puml"
        # output_filename = f"{directory_path}puml/{cim_type}{current_time}.puml"
        with open(output_filename, 'w') as f2:
            f2.write(pjson)

    print(f"RDF fails: {rdf_failcount}, JSON fails: {json_failcount}")
    with open(f"{directory_path}/failed_conversions.txt", 'w') as f:
        for line in json_fail_files:
            f.write(line)
